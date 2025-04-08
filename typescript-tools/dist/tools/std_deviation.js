"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateStandardDeviation = void 0;
const validation_1 = require("../utils/validation");
/**
 * Calculates the standard deviation of an array of numbers.
 *
 * @param numbers Array of numbers to calculate standard deviation for
 * @returns The standard deviation of the provided numbers
 */
function calculateStandardDeviation(numbers) {
    (0, validation_1.validateNumbers)(numbers);
    const mean = numbers.reduce((acc, num) => acc + num, 0) / numbers.length;
    const variance = numbers.reduce((acc, num) => acc + Math.pow(num - mean, 2), 0) / numbers.length;
    return Math.sqrt(variance);
}
exports.calculateStandardDeviation = calculateStandardDeviation;
