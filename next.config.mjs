/** @type {import('next').NextConfig} */

import path from 'path';
import { exec } from 'child_process';
import chokidar from 'chokidar'

const nextConfig = {
  output: 'export',
  basePath: '/Users/femibello/Documents/projects/bell-boy.github.io/out',
  webpack: (config, {dev, isServer}) => {
    if (dev && isServer) {
      const blogDir = path.resolve('blogs/')

      const watcher = chokidar.watch(blogDir, {
        persistent: true
      })

      watcher.on('change', () => {
        exec('npm run build:blogs', (err, stdout, stderr) => {
          if (err) {
            console.error(`error building blogs ${stderr}`)
          } else {
            console.log(`Blogs built successfully ${stdout}`)
          }
        })
      })
    }
    return config
  },
};

export default nextConfig;
