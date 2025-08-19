using System;
using System.IO;
using System.Linq;
using PSO_ANN.MODELS;
using PSO_ANN.ANN;
using PSO_ANN.UTILS;

namespace PSO_ANN
{
    class Program
    {
        static void Main(string[] args)
        {
            // File path setup
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string projDir = Path.GetFullPath(Path.Combine(exeDir, "..", "..", ".."));
            string csvPath = Path.Combine(projDir, "DATA", "housing.csv");
            if (!File.Exists(csvPath))
            {
                Console.WriteLine("Error: housing.csv not found.");
                return;
            }

            // Common settings
            int hiddenNeurons = 10;
            int particleCount = 50;
            int iterations = 1000;
            double learningRate = 0.01;
            double epsilon = 1e-4;

            // 1) PSO‑only optimization
            Console.WriteLine("--- PSO‑only Optimization ---");
            var (psoWeights, psoTrainMse, psoValMse) = PSOTrainer.Run(
                csvPath,
                hiddenNeurons,
                particleCount,
                iterations
            );
            Console.WriteLine($"PSO-only train MSE: {psoTrainMse:F6}");
            Console.WriteLine($"PSO-only val   MSE: {psoValMse:F6}");

            // 2) Hybrid PSO+GD optimization
            Console.WriteLine("--- Hybrid PSO+GD Optimization ---");
            var (hybWeights, hybTrainMse, hybValMse) = HybridTrainer.Run(
                csvPath,
                hiddenNeurons,
                particleCount,
                iterations,
                learningRate,
                epsilon
            );
            Console.WriteLine($"Hybrid train MSE: {hybTrainMse:F6}");
            Console.WriteLine($"Hybrid val   MSE: {hybValMse:F6}");

            // 3) First 5 validation predictions
            Console.WriteLine("--- First 5 Validation Predictions ---");
            var data = DataLoader.LoadAndNormalize(csvPath);
            var rnd = new Random(42);
            var valData = data.OrderBy(_ => rnd.Next())
                              .Skip((int)(0.8 * data.Count))
                              .Take(5)
                              .ToList();

            for (int i = 0; i < valData.Count; i++)
            {
                var (inputs, targets) = valData[i];

                var annPso = new NeuralNetwork(new[] { data[0].inputs.Length, hiddenNeurons, 1 });
                annPso.SetWeights(psoWeights);
                double predPso = annPso.Forward(inputs)[0];

                var annHyb = new NeuralNetwork(new[] { data[0].inputs.Length, hiddenNeurons, 1 });
                annHyb.SetWeights(hybWeights);
                double predHyb = annHyb.Forward(inputs)[0];

                // within 10% tolerance considered correct
                bool okPso = Math.Abs(predPso - targets[0]) <= 0.1 * targets[0];
                bool okHyb = Math.Abs(predHyb - targets[0]) <= 0.1 * targets[0];

                Console.WriteLine(
                    $"Sample {i + 1}: Actual={targets[0]:F3}, " +
                    $"PSO={predPso:F3} ({(okPso ? "✓" : "✗")}), " +
                    $"Hybrid={predHyb:F3} ({(okHyb ? "✓" : "✗")})"
                );
            }
        }
    }
    // hello this is an update
}
