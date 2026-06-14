import type { GpuContext } from './types';

const SHADER = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) coord : vec4f) -> @location(0) vec4f {
  let hdr = textureLoad(src, vec2u(coord.xy), 0).rgb;
  let mapped = hdr / (hdr + vec3f(1.0));     // Reinhard tonemap
  let srgb = pow(mapped, vec3f(1.0 / 2.2));  // gamma encode
  return vec4f(srgb, 1.0);
}
`;

// Maps the HDR scene target down to the 8-bit swapchain (tonemap + gamma).
export class Present {
  private readonly pipeline: GPURenderPipeline;
  private readonly layout: GPUBindGroupLayout;

  constructor(private readonly ctx: GpuContext) {
    const { device } = ctx;
    this.layout = device.createBindGroupLayout({
      label: 'present',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const module = device.createShaderModule({ label: 'present', code: SHADER });
    this.pipeline = device.createRenderPipeline({
      label: 'present',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.presentFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  encode(encoder: GPUCommandEncoder, src: GPUTextureView, dst: GPUTextureView): void {
    const bindGroup = this.ctx.device.createBindGroup({
      label: 'present',
      layout: this.layout,
      entries: [{ binding: 0, resource: src }],
    });
    const pass = encoder.beginRenderPass({
      label: 'present',
      colorAttachments: [{ view: dst, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }
}
