using PSO_ANN.ANN;
using PSO_ANN.MODELS;
using PSO_ANN.UTILS;
using System;
using System.IO;
using System.Linq;
using System.Diagnostics;

namespace PSO_ANN
{
    class Program
    {
        static void Main(string[] args)
        {
            // Locate dataset
            string exe = AppDomain.CurrentDomain.BaseDirectory;
            string proj = Path.GetFullPath(Path.Combine(exe, "..", "..", ".."));
            string csv = Path.Combine(proj, "DATA", "housing.csv");

            // Global settings
            int iterations = 1000;
            double[] targets = { 0.02, 0.01, 0.005 }; // pick sensible thresholds

            // ---- helper to run + report convergence ----
            void ReportConvergence(
                string label,
                Func<Action<int, double>, (double[] w, double train, double val)> run)
            {
                var sw = Stopwatch.StartNew();
                var firstHit = targets.ToDictionary(t => t, t => (int?)null);

                var result = run((iter, best) =>
                {
                    foreach (var t in targets)
                        if (best <= t && firstHit[t] == null)
                            firstHit[t] = iter;
                });

                sw.Stop();

                Console.WriteLine(
                    $"\n{label}  Train {result.train:F6} | Val {result.val:F6} | Total {sw.Elapsed.TotalSeconds:F2}s");

                foreach (var t in targets)
                {
                    int? k = firstHit[t];
                    string tt = k.HasValue
                        ? $"{k.Value} iters (~{sw.Elapsed.TotalMilliseconds * k.Value / iterations:F0} ms)"
                        : "not reached";
                    Console.WriteLine($"  Time-to-MSE ≤ {t}: {tt}");
                }
            }

            // PSO-only
            ReportConvergence("PSO", onIter =>
                PSOTrainer.Run(csv, iterations: iterations, onIter: onIter, logEvery: 1));

            // Hybrid PSO + GD (memetic)
            ReportConvergence("Hybrid PSO+GD", onIter =>
                HybridTrainer.Run(csv, iterations: iterations, onIter: onIter, logEvery: 1));

            if (Debugger.IsAttached)
            {
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
            }
        }
    }
}
