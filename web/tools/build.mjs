// Build (no bundler): clean dist, tsc the app and the bench, then copy pages, every src/**/*.css, assets/** and three's
// build files (dist/vendor/, loaded through the import map). Node standard library only.
import { cpSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), ".."), dist = join(web, "dist");
const tsc = (...args) => execFileSync(process.execPath, [join(web, "node_modules/typescript/bin/tsc"), ...args], { cwd: web, stdio: "inherit" });
const copy = (from, to) => { mkdirSync(dirname(to), { recursive: true }); cpSync(from, to, { recursive: true }); };
const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]));

rmSync(dist, { recursive: true, force: true });
tsc();
tsc("-p", "bench");
for (const f of ["index.html", "plain.html"]) copy(join(web, f), join(dist, f));
for (const f of walk(join(web, "src")).filter((f) => f.endsWith(".css"))) copy(f, join(dist, relative(join(web, "src"), f)));
copy(join(web, "assets"), join(dist, "assets"));
for (const f of ["three.webgpu.js", "three.tsl.js", "three.core.js"]) copy(join(web, "node_modules/three/build", f), join(dist, "vendor", f));
copy(join(web, "bench/index.html"), join(dist, "bench/index.html"));
console.log(`built ${relative(process.cwd(), dist) || "dist"}`);
