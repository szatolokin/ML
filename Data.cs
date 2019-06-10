using System;
using System.IO;

using MathNet.Numerics.LinearAlgebra;

namespace ML
{
    namespace Data
    {
        public class Data
        {
            public int Size;

            public Matrix<double> X;
            public Matrix<double> Y;

            public Generator Gen = new Generator();

            public Data(string fileName)
            {
                LoadFromFile(fileName);
            }

            public Data(int size = 5000, bool gen = true)
            {
                Generate(size, gen);
            }

            public void LoadFromFile(string fileName)
            {
                StreamReader file = new StreamReader(fileName);

                Size = Convert.ToInt32(file.ReadLine());

                X = Matrix<double>.Build.Dense(Size, 1);
                Y = Matrix<double>.Build.Dense(Size, 1);

                for (int i = 0; i < Size; ++i)
                {
                    X[i, 0] = Convert.ToDouble(file.ReadLine());
                    Y[i, 0] = Convert.ToDouble(file.ReadLine());
                }

                file.Close();
            }

            public void Generate(int size = 5000, bool gen = true)
            {
                Size = size;

                X = Matrix<double>.Build.Dense(Size, 1);
                Y = Matrix<double>.Build.Dense(Size, 1);

                if (gen)
                {
                    Random rand = new Random();

                    for (int i = 0; i < Size; ++i)
                    {
                        X[i, 0] = Gen.MinArg + rand.NextDouble() * (Gen.MaxArg - Gen.MinArg);
                        Y[i, 0] = Generator.Func(X[i, 0]);
                    }
                }
            }

            public void SaveToFile(string fileName)
            {
                StreamWriter file = new StreamWriter(fileName);

                file.WriteLine(Size);

                for (int i = 0; i < Size; ++i)
                {
                    file.WriteLine(X[i, 0]);
                    file.WriteLine(Y[i, 0]);
                }

                file.Close();
            }

            public Data[] Split(int size)
            {
                Data[] res = new Data[2];

                res[0] = new Data(size, false);
                res[1] = new Data(Size - size, false);

                Random rand = new Random();

                bool[] check = new bool[Size];

                for (int i = 0; i < size; ++i)
                {
                    int ind = rand.Next(Size);

                    while (check[ind])
                    {
                        ind = rand.Next(Size);
                    }

                    check[ind] = true;

                    res[0].X[i, 0] = X[ind, 0];
                    res[0].Y[i, 0] = Y[ind, 0];
                }

                int j = 0;

                for (int i = 0; i < Size; ++i)
                {
                    if (!check[i])
                    {
                        res[1].X[j, 0] = X[i, 0];
                        res[1].Y[j, 0] = Y[i, 0];

                        ++j;
                    }
                }

                return res;
            }
        }

        public class Generator
        {
            public double MinArg = 0.2;
            public double MaxArg = 3;

            public static double Func(double arg)
            {
                return Math.Log(arg);
            }
            /*
            public double MinArg = -Math.PI;
            public double MaxArg = 1;

            public static double Func(double arg)
            {
                return Math.Exp(arg);
            }
            */
        }
    }
}

/*
 * Data:
 * - создавать из файла
 * - создавать с генерацией
 * - создавать пустую
 * - изменять из файла
 * - изменять генерацией
 * - изменять обнулять
 * - случайное разбиение на две выборки размерами size и Size - size
 * 
 * Generator:
 * - отрезок аргументов для генерации
 * - исходная функция зависимости
*/
