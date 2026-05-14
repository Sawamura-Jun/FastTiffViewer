using System.Diagnostics;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;

const string PipeName = "FastTiffViewer.Singleton.Main";

string pathArg = args.Length > 0 ? args[0].Trim().Trim('"') : "";

try
{
    using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.Out);
    pipe.Connect(30); // 常駐プロセスがいればすぐ繋がる
    NativeMethods.AllowSetForegroundWindow(NativeMethods.ASFW_ANY); // 既存プロセスからの前面表示をWindowsに許可
    using var writer = new StreamWriter(pipe, new UTF8Encoding(false)) { AutoFlush = true };
    writer.WriteLine(string.IsNullOrWhiteSpace(pathArg) ? "PING" : $"OPEN\t{pathArg}");
    return;
}
catch
{
    // 常駐なし: 本体を起動
    string baseDir = AppContext.BaseDirectory;
    string viewerExe = Path.Combine(baseDir, "FastTiffViewer.exe");
    if (!File.Exists(viewerExe))
    {
        Console.Error.WriteLine("FastTiffViewer.exe not found.");
        Environment.Exit(1);
    }

    var psi = new ProcessStartInfo(viewerExe) { UseShellExecute = true };
    if (!string.IsNullOrWhiteSpace(pathArg))
    {
        psi.ArgumentList.Add(pathArg);
    }
    Process.Start(psi);
}

internal static partial class NativeMethods
{
    internal const int ASFW_ANY = -1;

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    internal static extern bool AllowSetForegroundWindow(int dwProcessId);
}
