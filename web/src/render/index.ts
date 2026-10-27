// Public API of the render module (see README.md).
export { createDeck, DEFAULT_PARAMS } from "./deck.js";
export type { Deck, DeckParams, Pick, RGB } from "./deck.js";
export { NEURONS, LAYERS, PER_LAYER, THRESHOLDS, rarityTableFromSamples, rarityF, levelOf, cellLevel } from "./rarity.js";
export type { RarityTable } from "./rarity.js";
export { DEFAULT_VIEW, neuronPos, labelPos } from "./geometry.js";
export type { View } from "./geometry.js";
export { makeSynth, recordedRarityTable, recordedQuantile, RECORDED_QUANTILES } from "./synth.js";
