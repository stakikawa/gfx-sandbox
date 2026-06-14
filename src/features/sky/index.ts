import { FRAME_WGSL, type Frame } from '../../core/frame';
import type { GpuContext } from '../../core/types';

const SHADER = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame : Frame;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) ndc : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o : VsOut;
  o.clip = vec4f(p[i], 1.0, 1.0);
  o.ndc = p[i];
  return o;
}

// Procedural sky: gradient + a bright HDR sun. dir is the world-space view ray.
fn sky(dir : vec3f) -> vec3f {
  let up = clamp(dir.y, 0.0, 1.0);
  let horizon = vec3f(0.62, 0.74, 0.92);
  let zenith = vec3f(0.18, 0.40, 0.82);
  var col = mix(horizon, zenith, pow(up, 0.45));

  // below the horizon, fade to a dim ground haze
  let ground = vec3f(0.30, 0.29, 0.27);
  col = mix(col, ground, clamp(-dir.y * 4.0, 0.0, 1.0));

  // sun disk + glow
  let s = max(dot(dir, frame.sunDir), 0.0);
  let disk = pow(s, 3000.0) * 50.0;
  let glow = pow(s, 8.0) * 0.5;
  col += (disk + glow) * vec3f(1.0, 0.95, 0.85);

  return col * frame.sunIntensity;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4f {
  // reconstruct the world-space ray from the near-plane point at this pixel
  let near = frame.invProj * vec4f(in.ndc, 0.0, 1.0);
  let dirView = normalize(near.xyz / near.w);
  let dir = normalize((frame.invView * vec4f(dirView, 0.0)).xyz);
  return vec4f(sky(dir), 1.0);
}
`;

// Fullscreen procedural sky background. Drawn before terrain; writes no depth.
export class Sky {
  private readonly pipeline: GPURenderPipeline;

  constructor(ctx: GpuContext, private readonly frame: Frame) {
    const module = ctx.device.createShaderModule({ label: 'sky', code: SHADER });
    this.pipeline = ctx.device.createRenderPipeline({
      label: 'sky',
      layout: ctx.device.createPipelineLayout({ bindGroupLayouts: [frame.bindGroupLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
      // shares the scene's depth attachment but never occludes / writes it
      depthStencil: { format: ctx.depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
    });
  }

  draw(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.frame.bindGroup);
    pass.draw(3);
  }
}
