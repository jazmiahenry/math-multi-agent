"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateMedian = void 0;
const error_simulator_1 = require("../utils/error_simulator");
const validation_1 = require("../utils/validation");
/**
 * Calculates the median of an array of numbers.
 * Can be configured to randomly introduce errors for testing purposes.
 *
 * @param numbers Array of numbers to calculate median for
 * @param errorConfig Optional error simulation configuration
 * @returns The median of the provided numbers (or an incorrect result if simulating errors)
 */
function calculateMedian(numbers, errorConfig) {
    // Validate input numbers first
    (0, validation_1.validateNumbers)(numbers);
    // Default error config - can be overridden by caller
    const defaultErrorConfig = {
        throwErrorRate: 0.2,
        incorrectResultRate: 0.2,
        incorrectResultFn: () => {
            // Custom incorrect result function specific to medians
            // Ways to generate wrong medians:
            // 1. Return mean instead of median
            // 2. Forget to handle even-length arrays correctly
            // 3. Use wrong index for the median
            // 4. Sort in descending order instead of ascending
            const errorType = Math.floor(Math.random() * 4);
            switch (errorType) {
                case 0:
                    // Return mean instead of median
                    return numbers.reduce((acc, num) => acc + num, 0) / numbers.length;
                case 1:
                    // Forget to handle even-length arrays correctly (just take one element)
                    {
                        const sorted = [...numbers].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        return sorted[mid]; // Always take the higher of the two middle elements
                    }
                case 2:
                    // Use wrong index for the median (off by one)
                    {
                        const sorted = [...numbers].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        if (sorted.length % 2 === 0) {
                            // For even-length arrays, take the wrong pair of elements
                            return (sorted[mid] + sorted[mid + 1]) / 2;
                        }
                        else {
                            // For odd-length arrays, take the element before or after median
                            return sorted[mid + (Math.random() > 0.5 ? 1 : -1)] || sorted[mid];
                        }
                    }
                case 3:
                    // Sort in descending order instead of ascending
                    {
                        const sorted = [...numbers].sort((a, b) => b - a); // Descending
                        const mid = Math.floor(sorted.length / 2);
                        if (sorted.length % 2 === 0) {
                            return (sorted[mid - 1] + sorted[mid]) / 2;
                        }
                        else {
                            return sorted[mid];
                        }
                    }
                default:
                    // Compute the correct median for fallback
                    const sorted = [...numbers].sort((a, b) => a - b);
                    const mid = Math.floor(sorted.length / 2);
                    if (sorted.length % 2 === 0) {
                        return (sorted[mid - 1] + sorted[mid]) / 2;
                    }
                    else {
                        return sorted[mid];
                    }
            }
        },
        silentError: false
    };
    // Combine default config with any user-provided config
    const finalErrorConfig = Object.assign(Object.assign({}, defaultErrorConfig), errorConfig);
    // Calculate median with error simulation
    return (0, error_simulator_1.simulateError)(() => {
        const sorted = [...numbers].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2;
        }
        else {
            return sorted[mid];
        }
    }, finalErrorConfig);
}
exports.calculateMedian = calculateMedian;
