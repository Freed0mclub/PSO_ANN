using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PSO_ANN.PSO
{
    public class Particle
    {
        public double[] Position { get; set; }
        public double[] Velocity { get; set; }
        public double[] BestPosition { get; set; }
        public double BestFitness { get; set; }

        private static Random rng = new Random();

        public Particle(int dimensions)
        {
            Position = new double[dimensions];
            Velocity = new double[dimensions];
            BestPosition = new double[dimensions];

            for (int i = 0; i < dimensions; i++)
            {
                Position[i] = rng.NextDouble() * 2 - 1;
                Velocity[i] = rng.NextDouble() * 0.2 - 0.1;
                BestPosition[i] = Position[i];
            }

            BestFitness = double.MaxValue;
        }
    }

    public class Swarm
    {
        public Particle[] Particles { get; private set; }
        public double[] GlobalBestPosition { get; private set; }
        public double GlobalBestFitness { get; private set; }

        private readonly double inertia = 0.729;
        private readonly double cognitive = 1.49445;
        private readonly double social = 1.49445;
        private static Random rng = new Random();

        public Swarm(int particleCount, int dimensions)
        {
            Particles = new Particle[particleCount];
            GlobalBestPosition = new double[dimensions];
            GlobalBestFitness = double.MaxValue;

            for (int i = 0; i < particleCount; i++)
                Particles[i] = new Particle(dimensions);
        }

        public void UpdateParticles(Func<double[], double> fitnessFunc)
        {
            foreach (var particle in Particles)
            {
                double fitness = fitnessFunc(particle.Position);

                if (fitness < particle.BestFitness)
                {
                    particle.BestFitness = fitness;
                    Array.Copy(particle.Position, particle.BestPosition, particle.Position.Length);
                }

                if (fitness < GlobalBestFitness)
                {
                    GlobalBestFitness = fitness;
                    Array.Copy(particle.Position, GlobalBestPosition, particle.Position.Length);
                }
            }

            foreach (var particle in Particles)
            {
                for (int d = 0; d < particle.Position.Length; d++)
                {
                    double rp = rng.NextDouble();
                    double rg = rng.NextDouble();

                    particle.Velocity[d] = inertia * particle.Velocity[d]
                        + cognitive * rp * (particle.BestPosition[d] - particle.Position[d])
                        + social * rg * (GlobalBestPosition[d] - particle.Position[d]);

                    particle.Position[d] += particle.Velocity[d];
                }
            }
        }
    }
}
