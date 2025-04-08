"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateMode = void 0;
const error_simulator_1 = require("../utils/error_simulator");
const validation_1 = require("../utils/validation");
/**
 * Calculates the mode(s) of an array of numbers.
 * Can be configured to randomly introduce errors for testing purposes.
 *
 * @param numbers Array of numbers to calculate mode for
 * @param errorConfig Optional error simulation configuration
 * @returns Array containing the mode(s) - the most frequently occurring value(s)
 */
function calculateMode(numbers, errorConfig) {
    // Validate input numbers first
    (0, validation_1.validateNumbers)(numbers);
    // Default error config - can be overridden by caller
    const defaultErrorConfig = {
        throwErrorRate: 0.2,
        incorrectResultRate: 0.2,
        incorrectResultFn: () => {
            // Custom incorrect result function specific to mode calculation
            // Ways to generate wrong modes:
            // 1. Return the median instead of mode(s)
            // 2. Return the mean instead of mode(s)
            // 3. Return the least frequent values instead of most frequent
            // 4. Only return one mode when there are multiple modes
            // 5. Return all values as modes
            const errorType = Math.floor(Math.random() * 5);
            switch (errorType) {
                case 0:
                    // Return median instead of mode
                    {
                        const sorted = [...numbers].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        if (sorted.length % 2 === 0) {
                            const median = (sorted[mid - 1] + sorted[mid]) / 2;
                            return [median];
                        }
                        else {
                            return [sorted[mid]];
                        }
                    }
                case 1:
                    // Return mean instead of mode
                    {
                        const mean = numbers.reduce((acc, num) => acc + num, 0) / numbers.length;
                        return [mean];
                    }
                case 2:
                    // Return least frequent values
                    {
                        const frequency = {};
                        // Count frequency of each number
                        for (const num of numbers) {
                            frequency[num] = (frequency[num] || 0) + 1;
                        }
                        // Find minimum frequency
                        let minFrequency = Infinity;
                        for (const num in frequency) {
                            minFrequency = Math.min(minFrequency, frequency[num]);
                        }
                        // Return all values with minimum frequency
                        const leastFrequent = [];
                        for (const num in frequency) {
                            if (frequency[num] === minFrequency) {
                                leastFrequent.push(Number(num));
                            }
                        }
                        return leastFrequent.sort((a, b) => a - b);
                    }
                case 3:
                    // Return only one mode when there are multiple
                    {
                        const frequency = {};
                        let maxFrequency = 0;
                        // Count frequency of each number
                        for (const num of numbers) {
                            frequency[num] = (frequency[num] || 0) + 1;
                            maxFrequency = Math.max(maxFrequency, frequency[num]);
                        }
                        // Find first value with maximum frequency
                        for (const num in frequency) {
                            if (frequency[num] === maxFrequency) {
                                return [Number(num)];
                            }
                        }
                        return []; // Should never reach here
                    }
                case 4:
                    // Return all values as modes
                    {
                        return [...new Set(numbers)].sort((a, b) => a - b);
                    }
                default:
                    // Compute the correct mode for fallback
                    const frequency = {};
                    let maxFrequency = 0;
                    for (const num of numbers) {
                        frequency[num] = (frequency[num] || 0) + 1;
                        maxFrequency = Math.max(maxFrequency, frequency[num]);
                    }
                    const modes = [];
                    for (const num in frequency) {
                        if (frequency[num] === maxFrequency) {
                            modes.push(Number(num));
                        }
                    }
                    return modes.sort((a, b) => a - b);
            }
        },
        silentError: false
    };
    // Combine default config with any user-provided config
    const finalErrorConfig = Object.assign(Object.assign({}, defaultErrorConfig), errorConfig);
    // Calculate mode with error simulation
    return (0, error_simulator_1.simulateError)(() => {
        // Create a frequency map of all values
        const frequency = {};
        let maxFrequency = 0;
        // Count occurrences of each number
        for (const num of numbers) {
            frequency[num] = (frequency[num] || 0) + 1;
            maxFrequency = Math.max(maxFrequency, frequency[num]);
        }
        // Find all values that occur with the maximum frequency
        const modes = [];
        for (const num in frequency) {
            if (frequency[num] === maxFrequency) {
                modes.push(Number(num));
            }
        }
        return modes.sort((a, b) => a - b);
    }, finalErrorConfig);
}
exports.calculateMode = calculateMode;
