// What the page may print of a token (one rule for every module): the chat template is real data (its frames and tape bars
// stay) but never text. Hidden: any special token `<|…|>` (`<|im_start|>`, `<|im_end|>`, …), `<think>` / `</think>` (the
// empty pair of non-thinking mode included), the role word right after `<|im_start|>` (user / assistant / system / tool), and
// whitespace-only tokens next to any of those (the template's newlines).

interface Toks { readonly tokens: readonly { text: string }[] }

const SPECIAL = /^(?:\s*(?:<\|[^<>|]*\|>|<\/?think>)\s*)+$/;

/** A template token by itself: a special token or a think tag (no context needed). */
export const isMarkup = (text: string) => SPECIAL.test(text);

function core(s: Toks, k: number): boolean {
  const t = s.tokens[k];
  if (!t) return false;
  if (isMarkup(t.text)) return true;
  const prev = s.tokens[k - 1];
  return !!prev && prev.text.trim() === "<|im_start|>" && /^\s*\w+\s*$/.test(t.text); // the role word
}

/** Token k is chat template (never printed). */
export function isTemplate(s: Toks, k: number): boolean {
  if (core(s, k)) return true;
  const t = s.tokens[k];
  return !!t && t.text.trim() === "" && (core(s, k - 1) || core(s, k + 1));
}

/** The text to print for token k: "" for template tokens. */
export const displayText = (s: Toks, k: number) => (isTemplate(s, k) ? "" : s.tokens[k]?.text ?? "");

/** Free text (e.g. the done frame's flushed tail) with any template markup removed. */
export const cleanText = (text: string) => text.replace(/<\|[^<>|]*\|>|<\/?think>/g, "");
