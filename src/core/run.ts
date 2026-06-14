import { mat4, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from './camera';
import { Renderer } from './device';
import { Frame } from './frame';
import { Present } from './present';
import type { FrameState, GpuContext, Scene } from './types';

type CreateScene = (ctx: GpuContext, frame: Frame, camera: OrbitCamera) => Scene;

// Boots the engine for a single scene page and drives the render loop. Each scene's
// main.ts calls this; the create callback wires the scene (and tweaks the camera).
export async function runScene(selector: string, create: CreateScene): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>(selector);
  if (!canvas) throw new Error(`canvas ${selector} not found`);

  if (!navigator.gpu) {
    canvas.replaceWith(
      Object.assign(document.createElement('p'), {
        textContent: 'WebGPU is not available in this browser.',
        style: 'color:#fff;font:14px sans-serif;padding:1rem',
      }),
    );
    throw new Error('WebGPU not supported');
  }
  const el: HTMLCanvasElement = canvas;

  const renderer = await Renderer.create(el);
  const camera = new OrbitCamera(el);
  const frame = new Frame(renderer.ctx.device);
  const present = new Present(renderer.ctx);
  const scene = create(renderer.ctx, frame, camera);

  window.addEventListener('resize', () => renderer.resize());

  const start = performance.now();

  const render = (): void => {
    const time = (performance.now() - start) / 1000;
    const view = camera.view();
    const proj = camera.proj(el.width / el.height);

    const state: FrameState = {
      view,
      proj,
      viewProj: mat4.multiply(proj, view),
      invView: mat4.inverse(view),
      invProj: mat4.inverse(proj),
      cameraPos: camera.position(),
      time,
      resolution: [el.width, el.height],
      sunDir: vec3.normalize(vec3.create(0.3, 0.5, 0.4)),
      sunIntensity: 1,
    };
    frame.update(state);

    const encoder = renderer.ctx.device.createCommandEncoder();
    scene.encode(encoder, state, { color: renderer.hdrView, depth: renderer.depthView });
    present.encode(encoder, renderer.hdrView, renderer.presentView);
    renderer.ctx.device.queue.submit([encoder.finish()]);

    requestAnimationFrame(render);
  };

  requestAnimationFrame(render);
}
