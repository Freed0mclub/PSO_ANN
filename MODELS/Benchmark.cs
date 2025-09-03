using System;
using System.Diagnostics;
using PSO_ANN.PSO;
using PSO_ANN.UTILS;

namespace PSO_ANN.MODELS
{
    
    /// Classic black-box benchmark functions and thin drivers to compare
    /// PSO vs Hybrid PSO+GD without involving a neural network.
    
    public static class Benchmark
    {
        // ---------- Benchmark functions ----------
        public static double Sphere(double[] x)
        {
            double s = 0.0; for (int i = 0; i < x.Length; i++) s += x[i] * x[i]; return s;
        }

        public static double Rastrigin(double[] x)
        {
            double A = 10.0; double s = A * x.Length;
            for (int i = 0; i < x.Length; i++)
                s += x[i] * x[i] - A * Math.Cos(2 * Math.PI * x[i]);
            return s;
        }

        public static double Rosenbrock(double[] x)
        {
            double s = 0.0;
            for (int i = 0; i < x.Length - 1; i++)
            {
                double a = x[i + 1] - x[i] * x[i];
                double b = 1 - x[i];
                s += 100 * a * a + b * b;
            }
            return s;
        }

        public static double SchafferF6(double[] x)
        {
            if (x.Length != 2) throw new ArgumentException("Schaffer F6 is 2-D");
            double r2 = x[0] * x[0] + x[1] * x[1];
            double sr = Math.Sqrt(r2);
            double s = Math.Sin(sr);
            double num = s * s - 0.5;
            double den = Math.Pow(1.0 + 0.001 * r2, 2.0);
            return 0.5 + num / den;
        }

        // ---------- Thin drivers ----------
        public static (double[] xbest, double fbest, TimeSpan elapsed) RunPSO(
            Func<double[], double> f,
            int dim,
            int iterations = 1000,
            int particles = 40,
            double inertia = 0.729,
            double c1 = 1.49445,
            double c2 = 1.49445,
            string? csvPath = null,
            int logEvery = 1)
        {
            var swarm = new Swarm(particles, dim)
            {
                Inertia = inertia,
                CognitiveFactor = c1,
                SocialFactor = c2
            };

            CsvLogger? logger = csvPath != null
                ? new CsvLogger(csvPath, "iter", "best_f", "elapsed_ms")
                : null;

            var sw = Stopwatch.StartNew();
            for (int it = 1; it <= iterations; it++)
            {
                swarm.UpdateParticles(f);
                if (logger != null && (it % logEvery == 0 || it == 1))
                    logger.WriteRow(it, swarm.GlobalBestFitness, sw.Elapsed.TotalMilliseconds);
            }
            sw.Stop();
            logger?.Dispose();

            return (Copy(swarm.GlobalBestPosition), swarm.GlobalBestFitness, sw.Elapsed);
        }

        public static (double[] xbest, double fbest, TimeSpan elapsed) RunHybrid(
            Func<double[], double> f,
            Func<double[], double[]> NumericalGrad, // numerial gradient
            int dim,
            int iterations = 1000,
            int particles = 40,
            int gdPeriod = 10,
            int gdElite = 5,
            int gdSteps = 3,
            double gdLR = 0.01,
            string? csvPath = null,
            int logEvery = 1)
        {
            var swarm = new HybridSwarm(particles, dim)
            {
                Inertia = 0.729,
                CognitiveFactor = 1.49445,
                SocialFactor = 1.49445,
                GDPeriod = gdPeriod,
                GDEliteCount = gdElite,
                GDSteps = gdSteps,
                GDLearningRate = gdLR
            };

            CsvLogger? logger = csvPath != null
                ? new CsvLogger(csvPath, "iter", "best_f", "elapsed_ms")
                : null;

            var sw = Stopwatch.StartNew();
            for (int it = 1; it <= iterations; it++)
            {
                swarm.UpdateParticlesHybrid(f, NumericalGrad);
                if (logger != null && (it % logEvery == 0 || it == 1))
                    logger.WriteRow(it, swarm.GlobalBestFitness, sw.Elapsed.TotalMilliseconds);
            }
            sw.Stop();
            logger?.Dispose();

            return (Copy(swarm.GlobalBestPosition), swarm.GlobalBestFitness, sw.Elapsed);
        }

        private static double[] Copy(double[] v)
        {
            var y = new double[v.Length];
            Array.Copy(v, y, v.Length);
            return y;
        }
    }
}
