import type { GpuContext } from './types';

const SIZE = 2048;
const FORMAT: GPUTextureFormat = 'depth32float';

// Sun shadow map: a depth render of the scene from the light, sampled by lit surfaces.
// The light view-projection lives in the frame UBO (frame.sunViewProj); this owns the
// depth target + comparison sampler and hands out the depth-only pass casters render into.
export class Shadow {
  static readonly size = SIZE;
  static readonly format = FORMAT;

  readonly depthView: GPUTextureView;
  readonly sampler: GPUSampler;

  constructor(ctx: GpuContext) {
    const tex = ctx.device.createTexture({
      label: 'shadow-map',
      size: [SIZE, SIZE],
      format: FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthView = tex.createView();
    this.sampler = ctx.device.createSampler({ label: 'shadow-cmp', compare: 'less' });
  }

  // Depth-only pass that shadow casters render into, from the sun's point of view.
  beginPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      label: 'shadow',
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
  }
}
