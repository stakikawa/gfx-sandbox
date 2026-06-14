import { mat4, vec3 } from 'wgpu-matrix';
import { OrbitCamera } from '../core/camera';
import { Renderer } from '../core/device';
import { Frame } from '../core/frame';
import { Present } from '../core/present';
import type { FrameState } from '../core/types';
import { HelloScene } from '../scenes/hello';

// Application entry point: wires core + the hello scene and drives the render loop.

const canvas = document.querySelector<HTMLCanvasElement>('#app');
if (!canvas) throw new Error('canvas #app not found');

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
const scene = new HelloScene(renderer.ctx, frame);
const present = new Present(renderer.ctx);

window.addEventListener('resize', () => renderer.resize());

const start = performance.now();

function render(): void {
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
    sunDir: vec3.normalize(vec3.create(0.3, 0.8, 0.4)),
    sunIntensity: 1,
  };
  frame.update(state);

  const encoder = renderer.ctx.device.createCommandEncoder();
  scene.encode(encoder, state, { color: renderer.hdrView, depth: renderer.depthView });
  present.encode(encoder, renderer.hdrView, renderer.presentView);
  renderer.ctx.device.queue.submit([encoder.finish()]);

  requestAnimationFrame(render);
}

requestAnimationFrame(render);
