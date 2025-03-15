import express from 'express';
import bodyParser from 'body-parser';
import { calculateMean } from './tools/mean';
import { calculateMedian } from './tools/median';
import { calculateMode } from './tools/mode';
import { calculateStandardDeviation } from './tools/std_deviation';
import { calculateProbabilityDistribution } from './tools/probability';
import { calculateEigen } from './tools/eigen';

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());

// Mean endpoint
app.post('/tool/mean', (req, res) => {
    try {
        const numbers: number[] = req.body.numbers;
        const result = calculateMean(numbers);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Median endpoint
app.post('/tool/median', (req, res) => {
    try {
        const numbers: number[] = req.body.numbers;
        const result = calculateMedian(numbers);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Mode endpoint
app.post('/tool/mode', (req, res) => {
    try {
        const numbers: number[] = req.body.numbers;
        const result = calculateMode(numbers);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Standard Deviation endpoint
app.post('/tool/std_deviation', (req, res) => {
    try {
        const numbers: number[] = req.body.numbers;
        const result = calculateStandardDeviation(numbers);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Probability Distribution endpoint
app.post('/tool/probability', (req, res) => {
    try {
        const frequencies: number[] = req.body.frequencies;
        const result = calculateProbabilityDistribution(frequencies);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Eigenvalues/Eigenvectors endpoint
app.post('/tool/eigen', (req, res) => {
    try {
        const matrix: number[][] = req.body.matrix;
        const result = calculateEigen(matrix);
        res.json({ result });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`TypeScript Tools Service is running on port ${port}`);
});