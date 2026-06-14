import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4, Vec3 } from 'wgpu-matrix';

const UP = vec3.create(0, 1, 0);

// Perspective camera orbiting a target: drag to rotate, wheel to zoom.
export class OrbitCamera {
  target = vec3.create(0, 0, 0);
  distance = 4;
  yaw = 0.7;
  pitch = 0.3;
  fov = Math.PI / 3;
  near = 0.1;
  far = 100;
  private dragging = false;

  constructor(canvas: HTMLCanvasElement) {
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
      this.yaw += e.movementX * 0.005;
      this.pitch += e.movementY * 0.005;
      const limit = Math.PI / 2 - 0.01;
      this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.distance = Math.max(0.5, Math.min(50, this.distance * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
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
