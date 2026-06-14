import { FRAME_WGSL, type Frame } from '../../core/frame';
import type { GpuContext } from '../../core/types';

const N = 200; // grid cells per side
const HALF = 16; // half-extent in world units
const AMP = 8; // peak-to-trough height range

// --- value-noise fbm heightfield, evaluated CPU-side ---
const fract = (x: number): number => x - Math.floor(x);
const hash = (ix: number, iy: number): number => fract(Math.sin(ix * 127.1 + iy * 311.7) * 43758.5453123);
const smooth = (t: number): number => t * t * (3 - 2 * t);

function vnoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const a = hash(ix, iy);
  const b = hash(ix + 1, iy);
  const c = hash(ix, iy + 1);
  const d = hash(ix + 1, iy + 1);
  const ux = smooth(fx);
  const uy = smooth(fy);
  return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy;
}

function fbm(x: number, y: number): number {
  let v = 0;
  let amp = 1;
  let freq = 1;
  let norm = 0;
  for (let o = 0; o < 5; o++) {
    v += amp * vnoise(x * freq, y * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return v / norm; // 0..1
}

function heightAt(wx: number, wz: number): number {
  return (fbm(wx * 0.05 + 10, wz * 0.05 + 10) - 0.4) * AMP;
}

// Interleaved [pos.xyz, normal.xyz] vertices + a triangle index buffer for the grid.
function buildMesh(): { vertexData: Float32Array; indexData: Uint32Array } {
  const side = N + 1;
  const cell = (HALF * 2) / N;
  const vertexData = new Float32Array(side * side * 6);

  for (let j = 0; j < side; j++) {
    for (let i = 0; i < side; i++) {
      const wx = -HALF + i * cell;
      const wz = -HALF + j * cell;
      const wy = heightAt(wx, wz);
      // central-difference slope → heightfield normal normalize(-dh/dx, 1, -dh/dz)
      const hx = (heightAt(wx + cell, wz) - heightAt(wx - cell, wz)) / (2 * cell);
      const hz = (heightAt(wx, wz + cell) - heightAt(wx, wz - cell)) / (2 * cell);
      const inv = 1 / Math.hypot(hx, 1, hz);
      const o = (j * side + i) * 6;
      vertexData[o] = wx;
      vertexData[o + 1] = wy;
      vertexData[o + 2] = wz;
      vertexData[o + 3] = -hx * inv;
      vertexData[o + 4] = inv;
      vertexData[o + 5] = -hz * inv;
    }
  }

  const indexData = new Uint32Array(N * N * 6);
  let k = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const tl = j * side + i;
      const tr = tl + 1;
      const bl = tl + side;
      const br = bl + 1;
      indexData[k++] = tl;
      indexData[k++] = bl;
      indexData[k++] = tr;
      indexData[k++] = tr;
      indexData[k++] = bl;
      indexData[k++] = br;
    }
  }

  return { vertexData, indexData };
}

const SHADER = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame : Frame;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) normal : vec3f,
  @location(1) height : f32,
};

@vertex
fn vs(@location(0) pos : vec3f, @location(1) normal : vec3f) -> VsOut {
  var o : VsOut;
  o.clip = frame.viewProj * vec4f(pos, 1.0);
  o.normal = normal;
  o.height = pos.y;
  return o;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let slope = 1.0 - n.y; // 0 = flat, → 1 = vertical face

  // height/slope-based albedo: grass on gentle low ground, rock on steep faces, snow on peaks
  let grass = vec3f(0.20, 0.34, 0.13);
  let rock = vec3f(0.30, 0.27, 0.24);
  let snow = vec3f(0.90, 0.92, 0.96);
  var albedo = mix(grass, rock, smoothstep(0.25, 0.55, slope));
  let snowLine = smoothstep(2.5, 4.5, in.height) * (1.0 - smoothstep(0.45, 0.75, slope));
  albedo = mix(albedo, snow, snowLine);

  // direct sun — warm, matches the sky's sun color
  let sunCol = vec3f(1.0, 0.95, 0.85);
  let ndl = max(dot(n, frame.sunDir), 0.0);
  let direct = sunCol * (ndl * frame.sunIntensity);

  // hemispheric sky ambient: sky color from above, dim ground bounce from below
  let skyAmbient = vec3f(0.45, 0.55, 0.72);
  let groundAmbient = vec3f(0.18, 0.16, 0.13);
  let ambient = mix(groundAmbient, skyAmbient, n.y * 0.5 + 0.5);

  let color = albedo * (direct + ambient);
  return vec4f(color, 1.0);
}
`;

// Static heightmap terrain: a noise grid uploaded once, drawn with depth and basic sun lighting.
export class Terrain {
  private readonly pipeline: GPURenderPipeline;
  private readonly vertexBuffer: GPUBuffer;
  private readonly indexBuffer: GPUBuffer;
  private readonly indexCount: number;

  constructor(ctx: GpuContext, private readonly frame: Frame) {
    const { device } = ctx;
    const { vertexData, indexData } = buildMesh();
    this.indexCount = indexData.length;

    this.vertexBuffer = device.createBuffer({
      label: 'terrain-vertices',
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

    this.indexBuffer = device.createBuffer({
      label: 'terrain-indices',
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indexData);

    const module = device.createShaderModule({ label: 'terrain', code: SHADER });
    this.pipeline = device.createRenderPipeline({
      label: 'terrain',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frame.bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 24,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  draw(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.frame.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }
}
