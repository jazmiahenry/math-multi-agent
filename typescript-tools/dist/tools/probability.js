"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateProbabilityDistribution = void 0;
const validation_1 = require("../utils/validation");
/**
 * Calculates the probability distribution from an array of frequencies.
 *
 * @param frequencies Array of frequency values
 * @returns Array of probabilities (each value divided by the total)
 * @throws Error if the total frequency is zero
 */
function calculateProbabilityDistribution(frequencies) {
    (0, validation_1.validateNumbers)(frequencies);
    const total = frequencies.reduce((acc, value) => acc + value, 0);
    if (total === 0) {
        throw new Error("Total frequency must be greater than zero.");
    }
    return frequencies.map(value => value / total);
}
exports.calculateProbabilityDistribution = calculateProbabilityDistribution;
