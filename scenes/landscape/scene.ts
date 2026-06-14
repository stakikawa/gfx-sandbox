import type { Frame } from '../../src/core/frame';
import type { FrameState, FrameTargets, GpuContext, Scene } from '../../src/core/types';
import { Terrain } from '../../src/features/terrain';

// Placeholder background until a sky feature lands (linear HDR, tonemapped on present).
const SKY_CLEAR = { r: 0.5, g: 0.7, b: 0.95, a: 1 };

// Terrain landscape. Owns pass order; gains sky + grass as features are added.
export class LandscapeScene implements Scene {
  private readonly terrain: Terrain;

  constructor(ctx: GpuContext, frame: Frame) {
    this.terrain = new Terrain(ctx, frame);
  }

  encode(encoder: GPUCommandEncoder, _frame: FrameState, targets: FrameTargets): void {
    const pass = encoder.beginRenderPass({
      label: 'terrain',
      colorAttachments: [{ view: targets.color, loadOp: 'clear', storeOp: 'store', clearValue: SKY_CLEAR }],
      depthStencilAttachment: {
        view: targets.depth,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store',
      },
    });
    this.terrain.draw(pass);
    pass.end();
  }
}
