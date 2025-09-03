using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using PSO_ANN.MODELS;
using PSO_ANN.UTILS;
using PSO_ANN.ANN;

// for benchmarking against functions. 

namespace PSO_ANN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Paths
            string exe = AppDomain.CurrentDomain.BaseDirectory;
            string proj = Path.GetFullPath(Path.Combine(exe, "..", "..", ".."));
            string resultsDir = Path.Combine(proj, "RESULTS");
            Directory.CreateDirectory(resultsDir);

            // Common settings
            int iterations = 2000;  
            int particles = 60;
            int logEvery = 10;

            // Helper to run + log both methods
            void RunBoth(
                string name,
                int dim,
                Func<double[], double> f,
                Func<double[], double[]> grad,
                string psoCsv,
                string hybCsv,
                CsvLogger summary)
            {
                Console.WriteLine($"\n=== {name} (dim={dim}) ===");

                // PSO
                var sw1 = Stopwatch.StartNew();
                var (xPSO, fPSO, tPSO) = Benchmark.RunPSO(
                    f: f,
                    dim: dim,
                    iterations: iterations,
                    particles: particles,
                    csvPath: Path.Combine(resultsDir, psoCsv),
                    logEvery: logEvery
                );
                sw1.Stop();
                Console.WriteLine($"[PSO]   f*={fPSO:E6}  time={tPSO.TotalSeconds:F2}s");

                // Hybrid (PSO + GD with analytic gradient)
                var sw2 = Stopwatch.StartNew();
                var (xHYB, fHYB, tHYB) = Benchmark.RunHybrid(
                    f: f,
                    NumericalGrad: grad,
                    dim: dim,
                    iterations: iterations,
                    particles: particles,
                    gdPeriod: 10,
                    gdElite: 6,
                    gdSteps: 3,
                    gdLR: 0.01,
                    csvPath: Path.Combine(resultsDir, hybCsv),
                    logEvery: logEvery
                );
                sw2.Stop();
                Console.WriteLine($"[H-PSO] f*={fHYB:E6}  time={tHYB.TotalSeconds:F2}s");

                summary.WriteRow(name, dim, iterations, particles, "PSO", fPSO, tPSO.TotalSeconds);
                summary.WriteRow(name, dim, iterations, particles, "H-PSO", fHYB, tHYB.TotalSeconds);
            }

            // --------- Analytic gradients for benchmarks ---------

            // Sphere: f(x)=sum x_i^2,  ∇f=2x
            static double[] SphereGrad(double[] x)
            {
                var g = new double[x.Length];
                for (int i = 0; i < x.Length; i++) g[i] = 2.0 * x[i];
                return g;
            }

            // Rastrigin: f= A*n + sum [x_i^2 - A cos(2π x_i)],  ∇f= 2x + 2πA sin(2πx)
            static double[] RastriginGrad(double[] x)
            {
                const double A = 10.0;
                var g = new double[x.Length];
                for (int i = 0; i < x.Length; i++)
                    g[i] = 2.0 * x[i] + 2.0 * Math.PI * A * Math.Sin(2.0 * Math.PI * x[i]);
                return g;
            }

            // Rosenbrock (classic 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
            static double[] RosenbrockGrad(double[] x)
            {
                int n = x.Length;
                var g = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double gi = 0.0;
                    if (i < n - 1)
                        gi += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
                    if (i > 0)
                        gi += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
                    g[i] = gi;
                }
                return g;
            }

            // Schaffer F6 (2-D): f=0.5 + (sin(√r^2)^2-0.5)/(1+0.001 r^2)^2
            static double[] SchafferF6Grad(double[] x)
            {
                if (x.Length != 2) throw new ArgumentException("Schaffer F6 is 2-D");
                double x0 = x[0], x1 = x[1];
                double r2 = x0 * x0 + x1 * x1;
                double s = Math.Sqrt(r2) + 1e-12; // avoid division by zero
                double sin2s = Math.Sin(2.0 * s);  // 2 sin s cos s
                double gval = Math.Sin(s) * Math.Sin(s) - 0.5;
                double h = Math.Pow(1.0 + 0.001 * r2, 2.0);

                // df/dx = (sin(2s)*(x/s)*h - g * 0.004 * x * (1+0.001 r2)) / h^2
                double common1 = 1.0 / (h);
                double common2 = 0.004 * (1.0 + 0.001 * r2) / (h * h);

                double gx = sin2s * (x0 / s) * common1 - gval * common2 * x0;
                double gy = sin2s * (x1 / s) * common1 - gval * common2 * x1;

                return new[] { gx, gy };
            }

            // --------- Run all and summarize ---------
            using var summary = new CsvLogger(Path.Combine(resultsDir, "benchmark_summary.csv"),
                                              "problem", "dim", "iterations", "particles", "method", "fbest", "elapsed_s");

            RunBoth(
                name: "Sphere",
                dim: 30,
                f: Benchmark.Sphere,
                grad: SphereGrad,
                psoCsv: "sphere_pso.csv",
                hybCsv: "sphere_hybrid.csv",
                summary: summary
            );

            RunBoth(
                name: "Rastrigin",
                dim: 10,
                f: Benchmark.Rastrigin,
                grad: RastriginGrad,
                psoCsv: "rastrigin_pso.csv",
                hybCsv: "rastrigin_hybrid.csv",
                summary: summary
            );

            RunBoth(
                name: "Rosenbrock",
                dim: 10,
                f: Benchmark.Rosenbrock,
                grad: RosenbrockGrad,
                psoCsv: "rosenbrock_pso.csv",
                hybCsv: "rosenbrock_hybrid.csv",
                summary: summary
            );

            RunBoth(
                name: "SchafferF6",
                dim: 2,
                f: Benchmark.SchafferF6,
                grad: SchafferF6Grad,
                psoCsv: "schafferF6_pso.csv",
                hybCsv: "schafferF6_hybrid.csv",
                summary: summary
            );

            Console.WriteLine($"\nBenchmark logs and summary written to: {resultsDir}");
        }
    }
}





// for Training Neural Networks uncomment the below code and load a comma delimated .csv file in the "runPSO" section 

/*
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
*/