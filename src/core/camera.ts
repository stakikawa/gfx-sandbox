import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4, Vec3 } from 'wgpu-matrix';

const UP = vec3.create(0, 1, 0);

// Perspective camera orbiting a target: drag to rotate, ctrl-drag to pan, wheel to zoom.
export class OrbitCamera {
  target = vec3.create(0, 0, 0);
  distance = 4;
  yaw = 0.7;
  pitch = 0.3;
  fov = Math.PI / 3;
  near = 0.1;
  far = 100;
  private dragging = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', (e) => {
      this.dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      if (e.ctrlKey) {
        this.pan(e.movementX, e.movementY);
        return;
      }
      this.yaw += e.movementX * 0.005;
      this.pitch += e.movementY * 0.005;
      const limit = Math.PI / 2 - 0.01;
      this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
    });
    // ctrl-drag pans, which on macOS is also a secondary-click — suppress the context menu
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.distance = Math.max(0.5, Math.min(50, this.distance * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
  }

  // Translate the look-at target across the view plane; world motion tracks the cursor
  // at the target's depth, so panning feels consistent at any zoom.
  private pan(dx: number, dy: number): void {
    const forward = vec3.normalize(vec3.sub(this.target, this.position()));
    const right = vec3.normalize(vec3.cross(forward, UP));
    const up = vec3.cross(right, forward);
    const worldPerPixel = (2 * this.distance * Math.tan(this.fov / 2)) / this.canvas.clientHeight;
    this.target = vec3.addScaled(this.target, right, -dx * worldPerPixel);
    this.target = vec3.addScaled(this.target, up, dy * worldPerPixel);
  }

  position(): Vec3 {
    const cp = Math.cos(this.pitch);
    const dir = vec3.create(cp * Math.cos(this.yaw), Math.sin(this.pitch), cp * Math.sin(this.yaw));
    return vec3.addScaled(this.target, dir, this.distance);
  }

  view(): Mat4 {
    return mat4.lookAt(this.position(), this.target, UP);
  }

  proj(aspect: number): Mat4 {
    return mat4.perspective(this.fov, aspect, this.near, this.far);
  }
}
