using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Win32;
using Microsoft.WindowsAPICodePack.Dialogs;

namespace OCR_APP
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            AppDomain.CurrentDomain.UnhandledException += (s, e) => { };
            InitializeComponent();
#if DEBUG
            txtImagesPath.Text = @"D:\MSc\ocr_app\samples\latest2";
            txtAnswersPath.Text = @"D:\MSc\ocr_engine\answers.csv";
#endif
        }

        private void btnExit_Click(object sender, RoutedEventArgs e)
        {
            if (_process != null && !_process.HasExited)
                _process.Kill();
            Application.Current.Shutdown();
        }

        readonly ConcurrentBag<string> _results = new ConcurrentBag<string>();
        private bool ValidateData()
        {
            if (string.IsNullOrEmpty(txtAnswersPath.Text.Trim()) ||
                string.IsNullOrEmpty(txtImagesPath.Text.Trim()) ||
                !Directory.Exists(txtImagesPath.Text) ||
                !File.Exists(txtAnswersPath.Text) ||
                !txtAnswersPath.Text.EndsWith(".csv"))
            {
                MessageBox.Show("Please enter valid path", "Error", MessageBoxButton.OK, MessageBoxImage.Error, MessageBoxResult.OK, MessageBoxOptions.DefaultDesktopOnly);
                return false;
            }
            return true;
        }

        private async void btnOCR_Click(object sender, RoutedEventArgs e)
        {
            if (!ValidateData())
                return;
            btnOCR.IsEnabled = false;
            txtAnswersPath.IsEnabled = false;
            txtImagesPath.IsEnabled = false;
            progressBar.Value = 0;
            Mouse.OverrideCursor = Cursors.Wait;
            this.txtProgress.Text = "Initializing...";
            dataGrid.Cursor = Cursors.Arrow;
            btnExit.Cursor = Cursors.Arrow;
            dataGrid.ForceCursor = true;
            btnExit.ForceCursor = true;
            Mouse.UpdateCursor();
            btnBrowseAnswersFile.IsEnabled = false;
            btnBrowseImagesDirectory.IsEnabled = false;
            await Ocr().ContinueWith(s =>
            {

                btnOCR.IsEnabled = true;
                txtAnswersPath.IsEnabled = true;
                txtImagesPath.IsEnabled = true;
                btnBrowseAnswersFile.IsEnabled = true;
                btnBrowseImagesDirectory.IsEnabled = true;
                progressBar.Value = 0;
                txtProgress.Text = "";
                Mouse.OverrideCursor = Cursors.Arrow;
            }, TaskContinuationOptions.ExecuteSynchronously);

        }
        private Process _process;
        readonly string _desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        private async Task Ocr()
        {
            var imagesPath = txtImagesPath.Text;
            var answersPath = txtAnswersPath.Text;
            var args = $"\"{imagesPath}\" \"{answersPath}\"";
            var watch = new Stopwatch();
            var total = Directory.GetFiles(imagesPath).Length;
            progressBar.Maximum = total;
            var count = 0;
            dataGrid.Items.Clear();
            if (File.Exists(Path.Combine(_desktop, "Errors.csv")))
                File.Delete(Path.Combine(_desktop, "Errors.csv"));
            if (File.Exists(Path.Combine(_desktop, "Results.csv")))
                File.Delete(Path.Combine(_desktop, "Results.csv"));
            await Task.Run(() =>
                {
                    watch.Start();
                    var pInfo = new ProcessStartInfo
                    {
#if DEBUG
                        FileName = @"D:\MSc\ocr_engine\ocr_engine\ocr_engine\ocr_engine.exe",
#elif TRACE
                    FileName = Path.Combine(Environment.CurrentDirectory, "ocr_engine", "ocr_engine.exe"),
#endif
                        Arguments = args,
                        RedirectStandardError = true,
                        RedirectStandardInput = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    _process = new Process();
                    _process.EnableRaisingEvents = true;
                    _process.OutputDataReceived += (s, e) =>
                    {
                        var data = e.Data;
                        Debug.WriteLine(data);
                        if (string.IsNullOrEmpty(data))
                            return;
                        if (data.StartsWith("R"))
                        {

                            Interlocked.Increment(ref count);
                            Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Input,
                                new Action(() =>
                                {
                                    progressBar.Value = count;
                                    txtProgress.Text = (int) ((double) count / (double) total * 100) + "%";
                                    var result = new Result
                                    {
                                        RollNo = data.Split(',')[0].Split(':')[1].Trim(),
                                        Correct = data.Split(',')[1].Split(':')[1].Trim(),
                                        Wrong = data.Split(',')[2].Split(':')[1].Trim(),
                                        Missing = data.Split(',')[3].Split(':')[1].Trim(),
                                        Total = data.Split(',')[4].Split(':')[1].Trim()
                                    };
                                    dataGrid.Items.Add(result);
                                }));
                            _results.Add(data);
                        }
                        else
                        {
                            if (data.Contains(':'))
                                File.AppendAllText(Path.Combine(_desktop, "Errors.csv"), data.Split(',')[0]+Environment.NewLine);
                            Interlocked.Increment(ref count);
                            Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Input,
                                new Action(() =>
                                {
                                    progressBar.Value = count;
                                    txtProgress.Text = (int)((double)count / (double)total * 100) + "%";
                                }));
#if DEBUG
                            Debug.WriteLine(data);
#endif
                        }
                    };
                    _process.ErrorDataReceived += (s, e) =>
                    {
                        MessageBox.Show(e.Data, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    };
                    _process.StartInfo = pInfo;
                    try
                    {
                        _process.Start();
                        int procId = _process.Id;
                    }
                    catch (InvalidOperationException exception)
                    {
                        MessageBox.Show(exception.Message, "Error",
                            MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                    catch (Exception exception)
                    {
                        MessageBox.Show(exception.Message, "Error",
                            MessageBoxButton.OK, MessageBoxImage.Error);
                        throw new TaskCanceledException();
                    }
                    _process.BeginOutputReadLine();
                    using (var errReader = _process.StandardError)
                    {
                        var error = errReader.ReadToEnd();
                        if (!string.IsNullOrEmpty(error) && !error.Contains("UserWarning"))
                        {
                            MessageBox.Show(error, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                            throw new TaskCanceledException();
                        }
                    }
                    //_process.BeginErrorReadLine();

                    _process.WaitForExit();
#if DEBUG
                    Debug.WriteLine(watch.Elapsed);
                    Debug.WriteLine($"Total: {total} Processed: {count}");
#endif
                    watch.Stop();
                    Application.Current.Dispatcher.BeginInvoke(
                        DispatcherPriority.Normal,
                        new Action(() =>
                        {
                            var results = _results.Distinct().OrderBy(x => int.Parse(x.Split(',')[0].Split(':')[1].Trim()));
                            File.WriteAllLines(Path.Combine(_desktop, "Result.csv"),
                                new List<string>() {"Roll No,Correct,Wrong,Missing,Total"});
                            File.AppendAllLines(Path.Combine(_desktop, "Result.csv"), results.Select(s =>
                            {
                                var data = s.Split(',');
                                return data.Aggregate("", (current, s1) => current + s1.Split(':')[1].Trim() + ",");
                            }));
                        }));
                }).ContinueWith(a =>
                {
                    File.AppendAllLines(Path.Combine(_desktop, "performance.log"),
                        new List<string> { $"Completed in {watch.Elapsed.ToString("mm\\:ss\\.ff")} seconds",""});
                    MessageBox.Show($"Completed in {watch.Elapsed.ToString("mm\\:ss\\.ff")} seconds", "Done!",
                        MessageBoxButton.OK, MessageBoxImage.Information);

                }
                , TaskContinuationOptions.NotOnCanceled|TaskContinuationOptions.OnlyOnRanToCompletion);
        }
        private void btnBrowseImagesDirectory_Click(object sender, RoutedEventArgs e)
        {

            var dlg = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                AddToMostRecentlyUsedList = false,
                AllowNonFileSystemItems = false,
                EnsurePathExists = true,
                EnsureReadOnly = false,
                EnsureValidNames = true,
                Multiselect = false,
                ShowPlacesList = true
            };
            if (dlg.ShowDialog() == CommonFileDialogResult.Ok)
                txtImagesPath.Text = dlg.FileName;
        }
        private void btnBrowseAnswersFile_Click(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog
            {
                CheckFileExists = true,
                CheckPathExists = true,
                Filter = "Csv file (*.csv)|*.csv",
                Multiselect = false
            };
            var flag = fileDialog.ShowDialog();

            if (flag != null && flag.Value)
                txtAnswersPath.Text = fileDialog.FileName;
        }
    }
}
