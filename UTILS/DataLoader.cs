using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace PSO_ANN.UTILS
{
    /// <summary>
    /// Loads and preprocesses space-delimited CSV data, normalizing features and targets to [0,1].
    /// </summary>
    public static class DataLoader
    {
        /// <summary>
        /// Reads a space- (or whitespace-) delimited file where the last column is the target value.
        /// Returns normalized feature-target pairs.
        /// </summary>
        /// <param name="filePath">Absolute or relative path to the data file.</param>
        /// <returns>List of (inputs[], targets[]) tuples with values in [0,1].</returns>
        public static List<(double[] inputs, double[] targets)> LoadAndNormalize(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Dataset file not found: {filePath}");

            // Read lines, split on whitespace
            var raw = File.ReadAllLines(filePath)
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line => line
                    .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
                    .Select(token => double.Parse(token, CultureInfo.InvariantCulture))
                    .ToArray())
                .ToList();

            int featureCount = raw[0].Length - 1;
            int sampleCount = raw.Count;

            // Separate features and targets
            var features = new double[sampleCount][];
            var targets = new double[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                features[i] = raw[i].Take(featureCount).ToArray();
                targets[i] = raw[i][featureCount];
            }

            // Compute min/max for each feature
            var fMin = new double[featureCount];
            var fMax = new double[featureCount];
            for (int j = 0; j < featureCount; j++)
            {
                fMin[j] = features.Min(row => row[j]);
                fMax[j] = features.Max(row => row[j]);
            }

            // Compute target min/max
            double tMin = targets.Min();
            double tMax = targets.Max();

            // Normalize to [0,1]
            var normalized = new List<(double[] inputs, double[] targets)>(sampleCount);
            for (int i = 0; i < sampleCount; i++)
            {
                var inNorm = new double[featureCount];
                for (int j = 0; j < featureCount; j++)
                {
                    double range = fMax[j] - fMin[j];
                    inNorm[j] = range > 0 ? (features[i][j] - fMin[j]) / range : 0.0;
                }
                double tRange = tMax - tMin;
                double tNorm = tRange > 0 ? (targets[i] - tMin) / tRange : 0.0;

                normalized.Add((inNorm, new double[] { tNorm }));
            }

            return normalized;
        }
    }
}