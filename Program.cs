using PSO_ANN.ANN;
using PSO_ANN.MODELS;
using PSO_ANN.UTILS;
using System;
using System.IO;
using System.Linq;

namespace PSO_ANN
{
    class Program
    {
        static void Main(string[] args)
        {
            string exe = AppDomain.CurrentDomain.BaseDirectory;
            string proj = Path.GetFullPath(Path.Combine(exe, "..", "..", ".."));
            string csv = Path.Combine(proj, "DATA", "housing.csv");

            // PSO-only
            var (wPso, psoTrain, psoVal) = PSOTrainer.Run(csv);
            Console.WriteLine($"PSO-only   Train MSE: {psoTrain:F6}");
            Console.WriteLine($"PSO-only   Val   MSE: {psoVal:F6}");

            // PSO + GD hybrid (memetic)
            var (wHy, hyTrain, hyVal) = HybridTrainer.Run(
                csv, hiddenNeurons: 10, particleCount: 50, iterations: 1000,
                gdPeriod: 10, gdSteps: 3, gdElite: 5, gdLR: 0.01, gradBatch: 64, epsilon: 1e-4
            );
            Console.WriteLine($"Hybrid     Train MSE: {hyTrain:F6}");
            Console.WriteLine($"Hybrid     Val   MSE: {hyVal:F6}");
        }
    }
    
}
