// Minimal types for node's built-in test runner (the project has no @types/node).
declare module "node:test" { export function test(name: string, fn: () => void | Promise<void>): void; }
declare module "node:assert/strict" {
  const assert: { (v: unknown, msg?: string): void; ok(v: unknown, msg?: string): void; equal(a: unknown, b: unknown, msg?: string): void; deepEqual(a: unknown, b: unknown, msg?: string): void; throws(fn: () => unknown, msg?: string): void };
  export default assert;
}
declare module "node:fs" { export function readFileSync(path: URL | string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number }; }
