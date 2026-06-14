import { FRAME_WGSL, type Frame } from '../../src/core/frame';
import type { FrameState, FrameTargets, GpuContext, Scene } from '../../src/core/types';

const SHADER = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame : Frame;

struct VsOut {
  @builtin(position) pos : vec4f,
  @location(0) color : vec3f,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  let p = array<vec3f, 3>(vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(0.0, 1.0, 0.0));
  let c = array<vec3f, 3>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  var o : VsOut;
  o.pos = frame.viewProj * vec4f(p[i], 1.0);
  o.color = c[i];
  return o;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

// A single world-space triangle drawn through the camera — validates the pipeline.
export class HelloScene implements Scene {
  private readonly pipeline: GPURenderPipeline;

  constructor(ctx: GpuContext, private readonly frame: Frame) {
    const module = ctx.device.createShaderModule({ label: 'triangle', code: SHADER });
    this.pipeline = ctx.device.createRenderPipeline({
      label: 'triangle',
      layout: ctx.device.createPipelineLayout({ bindGroupLayouts: [frame.bindGroupLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  encode(encoder: GPUCommandEncoder, _frame: FrameState, targets: FrameTargets): void {
    const pass = encoder.beginRenderPass({
      label: 'triangle',
      colorAttachments: [
        {
          view: targets.color,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.frame.bindGroup);
    pass.draw(3);
    pass.end();
  }
}
