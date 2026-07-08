// Deterministic non-square synthetic test scene for websam golden fixtures.
// 640x427, flat background + 3 solid shapes. No randomness anywhere.
//
// Shape geometry (source-pixel coords) — keep in sync with generate.mjs:
//   - Big circle:  center (180, 210), r = 90   -> bbox [90, 120, 270, 300]  (orange #D08770)
//   - Rectangle:   x 380..550, y 120..320       -> bbox [380, 120, 550, 320] (green  #A3BE8C)
//   - Small circle: center (100, 60), r = 45    -> bbox [55, 15, 145, 105]   (blue   #5E81AC)
//   - Background: #2E3440
import sharp from "sharp";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "fixtures", "scene-640x427.png");
mkdirSync(dirname(out), { recursive: true });

const W = 640;
const H = 427;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <rect x="0" y="0" width="${W}" height="${H}" fill="#2E3440"/>
  <circle cx="180" cy="210" r="90" fill="#D08770"/>
  <rect x="380" y="120" width="170" height="200" fill="#A3BE8C"/>
  <circle cx="100" cy="60" r="45" fill="#5E81AC"/>
</svg>`;

await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(out);
console.log(`wrote ${out} (${W}x${H})`);
