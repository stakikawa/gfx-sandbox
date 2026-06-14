import { defineConfig } from 'vite';

export default defineConfig({
  base: './', // relative asset paths — robust under any GitHub Pages subpath
  build: {
    rollupOptions: {
      // One entry per page: the gallery plus each scene.
      input: {
        gallery: 'index.html',
        landscape: 'scenes/landscape/index.html',
        hello: 'scenes/hello/index.html',
      },
    },
  },
});
