import { FRAME_WGSL, type Frame } from '../../core/frame';
import { Shadow } from '../../core/shadow';
import type { GpuContext } from '../../core/types';

const N = 200; // grid cells per side
const HALF = 16; // half-extent in world units
const AMP = 14; // peak-to-trough height range

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

// Horizon-based sky occlusion, baked once. Samples the heightfield outward in several
// directions, takes the highest horizon angle in each, and averages the blocked sky.
function aoAt(wx: number, wz: number, h: number): number {
  const DIRS = 8;
  const STEPS = 4;
  const MAX_R = 4.0;
  let occ = 0;
  for (let d = 0; d < DIRS; d++) {
    const a = (d / DIRS) * Math.PI * 2;
    const dx = Math.cos(a);
    const dz = Math.sin(a);
    let maxSlope = 0;
    for (let s = 1; s <= STEPS; s++) {
      const r = (s / STEPS) * MAX_R;
      const dh = heightAt(wx + dx * r, wz + dz * r) - h;
      if (dh > 0) maxSlope = Math.max(maxSlope, dh / r);
    }
    occ += maxSlope / Math.sqrt(1 + maxSlope * maxSlope); // sin(horizon angle)
  }
  return 1 - occ / DIRS; // 1 = open sky, → 0 = occluded
}

// Interleaved [pos.xyz, normal.xyz] vertices + a triangle index buffer for the grid.
function buildMesh(): { vertexData: Float32Array; indexData: Uint32Array } {
  const side = N + 1;
  const cell = (HALF * 2) / N;
  const vertexData = new Float32Array(side * side * 7);

  for (let j = 0; j < side; j++) {
    for (let i = 0; i < side; i++) {
      const wx = -HALF + i * cell;
      const wz = -HALF + j * cell;
      const wy = heightAt(wx, wz);
      // central-difference slope → heightfield normal normalize(-dh/dx, 1, -dh/dz)
      const hx = (heightAt(wx + cell, wz) - heightAt(wx - cell, wz)) / (2 * cell);
      const hz = (heightAt(wx, wz + cell) - heightAt(wx, wz - cell)) / (2 * cell);
      const inv = 1 / Math.hypot(hx, 1, hz);
      const o = (j * side + i) * 7;
      vertexData[o] = wx;
      vertexData[o + 1] = wy;
      vertexData[o + 2] = wz;
      vertexData[o + 3] = -hx * inv;
      vertexData[o + 4] = inv;
      vertexData[o + 5] = -hz * inv;
      vertexData[o + 6] = aoAt(wx, wz, wy);
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

struct Debug {
  aoStrength : f32, // 1 = AO on, 0 = off (debug toggle)
};
@group(1) @binding(0) var<uniform> dbg : Debug;

@group(2) @binding(0) var shadowMap : texture_depth_2d;
@group(2) @binding(1) var shadowSamp : sampler_comparison;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) normal : vec3f,
  @location(1) height : f32,
  @location(2) ao : f32,
  @location(3) worldPos : vec3f,
};

@vertex
fn vs(@location(0) pos : vec3f, @location(1) normal : vec3f, @location(2) ao : f32) -> VsOut {
  var o : VsOut;
  o.clip = frame.viewProj * vec4f(pos, 1.0);
  o.normal = normal;
  o.height = pos.y;
  o.ao = ao;
  o.worldPos = pos;
  return o;
}

// Depth-only stage that renders the terrain from the sun into the shadow map.
@vertex
fn vsShadow(@location(0) pos : vec3f) -> @builtin(position) vec4f {
  return frame.sunViewProj * vec4f(pos, 1.0);
}

// PCF sun-shadow lookup. 1 = lit, 0 = fully shadowed.
fn shadowFactor(worldPos : vec3f) -> f32 {
  let lc = frame.sunViewProj * vec4f(worldPos, 1.0);
  let proj = lc.xyz / lc.w;
  let uv = vec2f(proj.x * 0.5 + 0.5, proj.y * -0.5 + 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
    return 1.0; // outside the shadow map → treat as lit
  }
  let compareDepth = proj.z - 0.0015; // depth bias against acne
  let texel = 1.0 / f32(${Shadow.size});
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let off = vec2f(f32(dx), f32(dy)) * texel;
      sum += textureSampleCompareLevel(shadowMap, shadowSamp, uv + off, compareDepth);
    }
  }
  return sum / 9.0;
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
  let snowLine = smoothstep(4.5, 7.5, in.height) * (1.0 - smoothstep(0.45, 0.75, slope));
  albedo = mix(albedo, snow, snowLine);

  // direct sun — warm, matches the sky's sun color
  let sunCol = vec3f(1.0, 0.95, 0.85);
  let ndl = max(dot(n, frame.sunDir), 0.0);
  let direct = sunCol * (ndl * frame.sunIntensity) * shadowFactor(in.worldPos);

  // hemispheric sky ambient, attenuated by baked occlusion (sky light blocked by terrain)
  let skyAmbient = vec3f(0.45, 0.55, 0.72);
  let groundAmbient = vec3f(0.18, 0.16, 0.13);
  let ambient = mix(groundAmbient, skyAmbient, n.y * 0.5 + 0.5) * mix(1.0, in.ao, dbg.aoStrength);

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
  private readonly debugBuffer: GPUBuffer;
  private readonly debugBind: GPUBindGroup;
  private readonly shadowPipeline: GPURenderPipeline;
  private readonly shadowConsumerBind: GPUBindGroup;

  constructor(ctx: GpuContext, private readonly frame: Frame, shadow: Shadow) {
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

    // debug: AO toggle (press "O"). Replace with a GUI knob once Tweakpane lands.
    this.debugBuffer = device.createBuffer({
      label: 'terrain-debug',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.debugBuffer, 0, new Float32Array([1, 0, 0, 0]));
    const debugLayout = device.createBindGroupLayout({
      label: 'terrain-debug',
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.debugBind = device.createBindGroup({
      label: 'terrain-debug',
      layout: debugLayout,
      entries: [{ binding: 0, resource: { buffer: this.debugBuffer } }],
    });
    let aoOn = true;
    window.addEventListener('keydown', (e) => {
      if (e.key === 'o' || e.key === 'O') {
        aoOn = !aoOn;
        device.queue.writeBuffer(this.debugBuffer, 0, new Float32Array([aoOn ? 1 : 0, 0, 0, 0]));
      }
    });

    // shadow map consumed at group 2: depth texture + comparison sampler
    const shadowLayout = device.createBindGroupLayout({
      label: 'terrain-shadow',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    this.shadowConsumerBind = device.createBindGroup({
      label: 'terrain-shadow',
      layout: shadowLayout,
      entries: [
        { binding: 0, resource: shadow.depthView },
        { binding: 1, resource: shadow.sampler },
      ],
    });

    const module = device.createShaderModule({ label: 'terrain', code: SHADER });
    this.pipeline = device.createRenderPipeline({
      label: 'terrain',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [frame.bindGroupLayout, debugLayout, shadowLayout],
      }),
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 28,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
    });

    // depth-only pipeline that renders the terrain into the sun's shadow map
    this.shadowPipeline = device.createRenderPipeline({
      label: 'terrain-shadow',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frame.bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vsShadow',
        buffers: [{ arrayStride: 28, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: Shadow.format,
        depthWriteEnabled: true,
        depthCompare: 'less',
        depthBias: 2,
        depthBiasSlopeScale: 3,
      },
    });
  }

  // Render terrain depth from the sun's POV (into Shadow's depth pass).
  drawShadow(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.shadowPipeline);
    pass.setBindGroup(0, this.frame.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  draw(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.frame.bindGroup);
    pass.setBindGroup(1, this.debugBind);
    pass.setBindGroup(2, this.shadowConsumerBind);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }
}
