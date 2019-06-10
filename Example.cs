using System.IO;

using ML.Data;
using ML.LinearRegression;

namespace Example
{
    class Program
    {
        static void Main()
        {
            Data data1 = new Data();
            data1.SaveToFile("sample1.txt");

            Data data2 = new Data("sample1.txt");

            Model model1 = new Model(data2);
            model1.SaveToFile("model1.txt");

            Model model2 = new Model("model1.txt");

            data2.Generate();
            data2.SaveToFile("sample2.txt");

            model2.Training(data2);
            model2.SaveToFile("model2.txt");

            StreamWriter file = new StreamWriter("result1.txt");

            file.WriteLine(model1.QualityFunc(data1));
            file.WriteLine(model1.CrossValid(data1));

            file.Close();

            file = new StreamWriter("result2.txt");

            file.WriteLine(model2.QualityFunc(data1));
            file.WriteLine(model2.CrossValid(data2));

            file.Close();
        }
    }
}

/*
 * Example:
 * - подключаем Data и LinearRegression
 * - генерируем выборку 1 по-умолчанию и сохраняем в файл
 * - создаем выборку 2 из сохраненной выборки 1
 * - создаем и обучаем модель 1 по выборке 2 и сохраняем ее в файл
 * - создаем модель 2 из файла модели 1
 * - повторно генерируем по-умолчанию выборку 2 и сохраняем в файл
 * - обучаем модель 2 по выборке 2 и сохраняем в файл
 * - вычисляем функционал потерь и скользящий контроль для модели 1 по выборке 1 и сохраняем в файл
 * - вычисляем функционал потерь модели 2 по выборке 1 и скользящий контроль модели 2 по выборке 2 и сохраняем в файл
*/

