import type { Frame } from '../../src/core/frame';
import type { FrameState, FrameTargets, GpuContext, Scene } from '../../src/core/types';
import { Sky } from '../../src/features/sky';
import { Terrain } from '../../src/features/terrain';

// Terrain landscape under a procedural sky. Owns pass order; gains grass etc. later.
export class LandscapeScene implements Scene {
  private readonly sky: Sky;
  private readonly terrain: Terrain;

  constructor(ctx: GpuContext, frame: Frame) {
    this.sky = new Sky(ctx, frame);
    this.terrain = new Terrain(ctx, frame);
  }

  encode(encoder: GPUCommandEncoder, _frame: FrameState, targets: FrameTargets): void {
    const pass = encoder.beginRenderPass({
      label: 'landscape',
      colorAttachments: [{ view: targets.color, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: {
        view: targets.depth,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store',
      },
    });
    this.sky.draw(pass); // background — no depth write
    this.terrain.draw(pass); // geometry over sky — depth tested
    pass.end();
  }
}
