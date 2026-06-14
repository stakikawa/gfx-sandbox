import type { GpuContext } from './types';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

// Owns the device, swapchain, and all screen-sized targets; recreates them on resize.
export class Renderer {
  readonly ctx: GpuContext;
  hdrView!: GPUTextureView;
  depthView!: GPUTextureView;
  private readonly context: GPUCanvasContext;
  private hdr!: GPUTexture;
  private depth!: GPUTexture;

  private constructor(ctx: GpuContext, context: GPUCanvasContext) {
    this.ctx = ctx;
    this.context = context;
    this.resize();
  }

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter available');
    const device = await adapter.requestDevice();

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Failed to acquire WebGPU canvas context');

    const presentFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: presentFormat, alphaMode: 'opaque' });

    return new Renderer(
      { device, canvas, presentFormat, hdrFormat: HDR_FORMAT, depthFormat: DEPTH_FORMAT },
      context,
    );
  }

  // View of the current swapchain image. Call once per frame.
  get presentView(): GPUTextureView {
    return this.context.getCurrentTexture().createView();
  }

  // Match the backing store and HDR target to the canvas's displayed size.
  resize(): void {
    const { device, canvas } = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (this.hdr && canvas.width === w && canvas.height === h) return;

    canvas.width = w;
    canvas.height = h;

    this.hdr?.destroy();
    this.hdr = device.createTexture({
      label: 'hdr-target',
      size: [w, h],
      format: this.ctx.hdrFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdr.createView();

    this.depth?.destroy();
    this.depth = device.createTexture({
      label: 'depth-target',
      size: [w, h],
      format: this.ctx.depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depth.createView();
  }
}
