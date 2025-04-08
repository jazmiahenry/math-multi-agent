"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const body_parser_1 = __importDefault(require("body-parser"));
const mean_1 = require("./tools/mean");
const median_1 = require("./tools/median");
const mode_1 = require("./tools/mode");
const std_deviation_1 = require("./tools/std_deviation");
const probability_1 = require("./tools/probability");
const eigen_1 = require("./tools/eigen");
const app = (0, express_1.default)();
const port = process.env.PORT || 3000;
app.use(body_parser_1.default.json());
// Mean endpoint
app.post('/tool/mean', (req, res) => {
    try {
        const numbers = req.body.numbers;
        const result = (0, mean_1.calculateMean)(numbers);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
// Median endpoint
app.post('/tool/median', (req, res) => {
    try {
        const numbers = req.body.numbers;
        const result = (0, median_1.calculateMedian)(numbers);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
// Mode endpoint
app.post('/tool/mode', (req, res) => {
    try {
        const numbers = req.body.numbers;
        const result = (0, mode_1.calculateMode)(numbers);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
// Standard Deviation endpoint
app.post('/tool/std_deviation', (req, res) => {
    try {
        const numbers = req.body.numbers;
        const result = (0, std_deviation_1.calculateStandardDeviation)(numbers);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
// Probability Distribution endpoint
app.post('/tool/probability', (req, res) => {
    try {
        const frequencies = req.body.frequencies;
        const result = (0, probability_1.calculateProbabilityDistribution)(frequencies);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
// Eigenvalues/Eigenvectors endpoint
app.post('/tool/eigen', (req, res) => {
    try {
        const matrix = req.body.matrix;
        const result = (0, eigen_1.calculateEigen)(matrix);
        res.json({ result });
    }
    catch (error) {
        res.status(400).json({ error: error.message });
    }
});
app.listen(port, () => {
    console.log(`TypeScript Tools Service is running on port ${port}`);
});
