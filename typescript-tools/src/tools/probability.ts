import { validateNumbers } from "../utils/validation";

/**
 * Calculates the probability distribution from an array of frequencies.
 * 
 * @param frequencies Array of frequency values
 * @returns Array of probabilities (each value divided by the total)
 * @throws Error if the total frequency is zero
 */
export function calculateProbabilityDistribution(frequencies: number[]): number[] {
    validateNumbers(frequencies);
    
    const total = frequencies.reduce((acc, value) => acc + value, 0);
    
    if (total === 0) {
        throw new Error("Total frequency must be greater than zero.");
    }
    
    return frequencies.map(value => value / total);
}