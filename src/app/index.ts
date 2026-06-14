// Application entry point. Acquires the canvas and confirms WebGPU is available.
// TODO: device/swapchain setup, scene registry, and render loop.

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

console.log('WebGPU available.');
