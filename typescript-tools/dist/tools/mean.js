"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateMean = void 0;
const error_simulator_1 = require("../utils/error_simulator");
const validation_1 = require("../utils/validation");
/**
 * Calculates the mean (average) of an array of numbers.
 * Can be configured to randomly introduce errors for testing purposes.
 *
 * @param numbers Array of numbers to calculate mean for
 * @param errorConfig Optional error simulation configuration
 * @returns The mean of the provided numbers (or an incorrect result if simulating errors)
 */
function calculateMean(numbers, errorConfig) {
    // Validate input numbers first
    (0, validation_1.validateNumbers)(numbers);
    // Default error config - can be overridden by caller
    const defaultErrorConfig = {
        throwErrorRate: 0.2,
        incorrectResultRate: 0.2,
        incorrectResultFn: () => {
            // Custom incorrect result function specific to means
            // Ways to generate wrong means:
            // 1. Return median instead of mean
            // 2. Forget to divide by length
            // 3. Return wrong mean by slightly altering the result
            const correctMean = numbers.reduce((acc, num) => acc + num, 0) / numbers.length;
            const errorType = Math.floor(Math.random() * 3);
            switch (errorType) {
                case 0:
                    // Return median instead of mean
                    const sorted = [...numbers].sort((a, b) => a - b);
                    const middle = Math.floor(sorted.length / 2);
                    return sorted.length % 2 === 0
                        ? (sorted[middle - 1] + sorted[middle]) / 2
                        : sorted[middle];
                case 1:
                    // Forget to divide by length (return sum)
                    return numbers.reduce((acc, num) => acc + num, 0);
                case 2:
                    // Return slightly wrong mean (off by random percentage)
                    const errorFactor = 1 + (Math.random() > 0.5 ? 0.1 : -0.1) * Math.random();
                    return correctMean * errorFactor;
                default:
                    return correctMean;
            }
        },
        silentError: false
    };
    // Combine default config with any user-provided config
    const finalErrorConfig = Object.assign(Object.assign({}, defaultErrorConfig), errorConfig);
    // Calculate mean with error simulation
    return (0, error_simulator_1.simulateError)(() => {
        const sum = numbers.reduce((acc, num) => acc + num, 0);
        return sum / numbers.length;
    }, finalErrorConfig);
}
exports.calculateMean = calculateMean;
