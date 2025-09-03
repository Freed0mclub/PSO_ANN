using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using PSO_ANN.MODELS;
using PSO_ANN.UTILS;
using PSO_ANN.ANN;

namespace PSO_ANN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // --- Paths ---
            string exe = AppDomain.CurrentDomain.BaseDirectory;
            string proj = Path.GetFullPath(Path.Combine(exe, "..", "..", ".."));
            string csv = Path.Combine(proj, "DATA", "housing.csv");

            // Put all logs here
            string resultsDir = Path.Combine(proj, "RESULTS");
            Directory.CreateDirectory(resultsDir);

            // --- Common trainer settings ---
            int iterations = 1000;
            int particles = 50;
            int hiddenNeurons = 10;
            int logEvery = 1;

            // --- Run PSO ---
            string psoLog = Path.Combine(resultsDir, "housing_pso.csv");
            Console.WriteLine($"[PSO] Logging to: {psoLog}");

            var swPSO = Stopwatch.StartNew();
            var (wPso, trainPso, valPso) = PSOTrainer.Run(
                csvPath: csv,
                hiddenNeurons: hiddenNeurons,
                particleCount: particles,
                iterations: iterations,
                logEvery: logEvery,
                logCsvPath: psoLog,
                onIter: (it, best) => { if (it % 100 == 0) Console.WriteLine($"[PSO] it={it} best={best:F6}"); }
            );
            swPSO.Stop();

            // --- Run Hybrid PSO + local GD ---
            string hybLog = Path.Combine(resultsDir, "housing_hybrid.csv");
            Console.WriteLine($"[H-PSO] Logging to: {hybLog}");

            var swHYB = Stopwatch.StartNew();
            var (wHy, trainHy, valHy) = HybridTrainer.Run(
                csvPath: csv,
                hiddenNeurons: hiddenNeurons,
                particleCount: particles,
                iterations: iterations,
                gdPeriod: 10, gdElite: 5, gdSteps: 3, gdLR: 0.01,
                logEvery: logEvery,
                logCsvPath: hybLog,
                onIter: (it, best) => { if (it % 100 == 0) Console.WriteLine($"[H-PSO] it={it} best={best:F6}"); }
            );
            swHYB.Stop();

            // --- Summary CSV ---
            using (var summary = new CsvLogger(Path.Combine(resultsDir, "summary.csv"),
                                               "method", "iterations", "particles", "hidden",
                                               "train_mse", "val_mse", "elapsed_s"))
            {
                summary.WriteRow("PSO", iterations, particles, hiddenNeurons, trainPso, valPso, swPSO.Elapsed.TotalSeconds);
                summary.WriteRow("H-PSO", iterations, particles, hiddenNeurons, trainHy, valHy, swHYB.Elapsed.TotalSeconds);
            }

            // --- Save first 5 validation predictions (for both models) ---
            // Load + split the dataset the same way the trainers do
            var all = DataLoader.LoadAndNormalize(csv);
            int nFeat = all[0].inputs.Length;
            var rng = new Random(42);
            var shuffled = all.OrderBy(_ => rng.Next()).ToList();
            int split = (int)(0.8 * shuffled.Count);
            var val = shuffled.Skip(split).ToList();           // validation slice

            var topo = new[] { nFeat, hiddenNeurons, 1 };

            // PSO predictions
            using (var preds = new CsvLogger(Path.Combine(resultsDir, "predictions_pso.csv"),
                                             "index", "y_true", "y_pred"))
            {
                var ann = new NeuralNetwork(topo);
                ann.SetWeights(wPso);

                for (int i = 0; i < Math.Min(5, val.Count); i++)
                {
                    var (x, t) = val[i];
                    var y = ann.Forward(x);
                    preds.WriteRow(i, t[0], y[0]);
                }
            }

            // Hybrid predictions
            using (var preds = new CsvLogger(Path.Combine(resultsDir, "predictions_hybrid.csv"),
                                             "index", "y_true", "y_pred"))
            {
                var ann = new NeuralNetwork(topo);
                ann.SetWeights(wHy);

                for (int i = 0; i < Math.Min(5, val.Count); i++)
                {
                    var (x, t) = val[i];
                    var y = ann.Forward(x);
                    preds.WriteRow(i, t[0], y[0]);
                }
            }

            // Console summary
            Console.WriteLine();
            Console.WriteLine($"PSO:   train={trainPso:F6}  val={valPso:F6}  time={swPSO.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"H-PSO: train={trainHy:F6}  val={valHy:F6}  time={swHYB.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"\nCSV logs written to: {resultsDir}");
        }
    }
}
