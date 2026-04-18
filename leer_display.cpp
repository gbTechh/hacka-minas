#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

const int X1=600, Y1=250, X2=1150, Y2=480;
const int UMBRAL_ROJO  = 70;
const int MIN_PIXELES  = 80;
const int GAP_FRAMES   = 200;
const int RESET_CAMION = 50;

struct Lectura {
    int frame;
    std::string tiempo;
    double tiempo_seg;
    int peso;
    double confianza;
};

// ── Preprocesa ROI: (R-B) → umbral → escala 5× → dilata ───────
cv::Mat preprocesar(cv::Mat& frame) {
    if (frame.rows < Y2 || frame.cols < X2) return {};
    cv::Mat roi = frame(cv::Range(Y1,Y2), cv::Range(X1,X2));
    std::vector<cv::Mat> ch; cv::split(roi, ch);
    cv::Mat rojo; cv::subtract(ch[2], ch[0], rojo);
    cv::Mat bin;
    cv::threshold(rojo, bin, UMBRAL_ROJO, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(bin) < MIN_PIXELES) return {};
    cv::Mat grande;
    cv::resize(bin, grande, cv::Size(), 5, 5, cv::INTER_NEAREST);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(grande, grande, k, cv::Point(-1,-1), 2);
    return grande;
}

int main(int argc, char** argv) {
    // Detectar directorio del script para encontrar ocr_helper.py
    std::string script_dir = ".";
    if (argc > 0) {
        std::string prog(argv[0]);
        auto pos = prog.rfind('/');
        if (pos != std::string::npos) script_dir = prog.substr(0, pos);
    }

    std::ifstream file("frames_display.txt");
    if (!file.is_open()) { std::cerr << "❌ No se puede abrir frames_display.txt\n"; return 1; }
    std::vector<int> frames;
    int f; while (file >> f) frames.push_back(f);
    file.close();

    cv::VideoCapture cap("videor.mp4");
    if (!cap.isOpened()) { std::cerr << "❌ No se puede abrir videor.mp4\n"; return 1; }
    double fps = cap.get(cv::CAP_PROP_FPS);

    // ── Fase 1: preprocesar frames y guardar imágenes en /tmp ──
    std::cout << "Fase 1: preprocesando " << frames.size() << " frames...\n";

    std::string ocr_input = "/tmp/ocr_input.txt";
    std::ofstream inp(ocr_input);

    std::vector<int> valid_frames;
    std::vector<std::string> img_paths;

    for (int fn : frames) {
        cap.set(cv::CAP_PROP_POS_FRAMES, fn);
        cv::Mat frame;
        if (!cap.read(frame)) continue;

        cv::Mat proc = preprocesar(frame);
        if (proc.empty()) continue;

        char path[64];
        snprintf(path, sizeof(path), "/tmp/display_%06d.png", fn);
        cv::imwrite(path, proc);

        inp << fn << " " << path << "\n";
        valid_frames.push_back(fn);
        img_paths.push_back(std::string(path));
    }
    inp.close();
    cap.release();

    std::cout << "  " << valid_frames.size() << " frames con señal de display.\n";

    // ── Fase 2: OCR con EasyOCR (Python) ──────────────────────
    std::cout << "Fase 2: OCR en " << valid_frames.size() << " frames...\n\n";

    // Buscar el entorno virtual adyacente al ejecutable o al directorio actual
    std::string venv_activate =
        "[ -f " + script_dir + "/venv_minas/bin/activate ] && "
        "source " + script_dir + "/venv_minas/bin/activate || "
        "source venv_minas/bin/activate 2>/dev/null";
    std::string cmd =
        "bash -c '" + venv_activate + " && "
        "python3 " + script_dir + "/ocr_helper.py < " + ocr_input +
        " 2>/dev/null'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { std::cerr << "❌ No se pudo lanzar ocr_helper.py\n"; return 1; }

    // Mapa frame → {valor, confianza}
    std::map<int, std::pair<int,double>> ocr_results;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        int fn; double val, conf;
        if (sscanf(buf, "%d %lf %lf", &fn, &val, &conf) == 3 && val > 0) {
            ocr_results[fn] = {(int)val, conf};
        }
    }
    pclose(pipe);

    // Limpiar temporales
    for (auto& p : img_paths) remove(p.c_str());
    remove(ocr_input.c_str());

    // ── Imprimir lecturas individuales ─────────────────────────
    double fps_local = fps;
    std::vector<Lectura> lecturas;
    for (int fn : valid_frames) {
        auto it = ocr_results.find(fn);
        if (it == ocr_results.end()) continue;
        auto [val, conf] = it->second;

        double seg = fn / fps_local;
        char tbuf[10];
        snprintf(tbuf, sizeof(tbuf), "%02d:%02d", (int)(seg/60), (int)seg%60);
        lecturas.push_back({fn, std::string(tbuf), seg, val, conf});

        std::cout << "  Frame " << std::setw(5) << fn
                  << " (" << tbuf << ") → " << val << "t"
                  << "  conf=" << std::fixed << std::setprecision(2) << conf << "\n";
    }

    if (lecturas.empty()) { std::cout << "❌ Sin lecturas\n"; return 0; }

    // ── Agrupar por proximidad temporal ───────────────────────
    std::vector<std::vector<Lectura>> grupos;
    std::vector<Lectura> grupo = {lecturas[0]};
    for (size_t i = 1; i < lecturas.size(); i++) {
        if (lecturas[i].frame - grupo.back().frame < GAP_FRAMES)
            grupo.push_back(lecturas[i]);
        else { grupos.push_back(grupo); grupo = {lecturas[i]}; }
    }
    grupos.push_back(grupo);

    // ── Tabla resumen ──────────────────────────────────────────
    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << std::setw(4) << "#"
              << std::setw(9) << "Tiempo"
              << std::setw(7) << "Peso"
              << std::setw(7) << "Conf"
              << std::setw(8) << "Delta"
              << std::setw(9) << "Lecturas" << "\n";
    std::cout << std::string(55, '=') << "\n";

    std::ofstream csv("cargas_cpp.csv");
    csv << "frame,tiempo,tiempo_seg,peso_t,confianza\n";

    int peso_prev=0, peso_max=0, total=0;
    double delta_sum=0;

    for (size_t i = 0; i < grupos.size(); i++) {
        auto& g = grupos[i];
        std::map<int,int> cnt;
        for (auto& l : g) cnt[l.peso]++;
        int peso = std::max_element(cnt.begin(), cnt.end(),
            [](auto& a, auto& b){ return a.second < b.second; })->first;
        Lectura mejor = g[0];
        for (auto& l : g)
            if (l.peso == peso && l.confianza > mejor.confianza) mejor = l;

        int delta = peso - peso_prev;
        bool es_nuevo = (peso_prev > 0 && peso < peso_prev - RESET_CAMION);
        if (es_nuevo) {
            std::cout << "\n  🚛 NUEVO CAMIÓN detectado\n\n";
            peso_prev = 0; delta = peso;
        }
        char signo = (delta >= 0) ? '+' : '-';
        std::cout << "  #" << std::setw(2) << (i+1)
                  << "   " << mejor.tiempo
                  << "   " << std::setw(4) << peso << "t"
                  << "   " << std::fixed << std::setprecision(2) << mejor.confianza
                  << "  " << signo << std::setw(3) << std::abs(delta) << "t"
                  << "   " << g.size() << " frames\n";

        csv << mejor.frame << "," << mejor.tiempo << ","
            << mejor.tiempo_seg << "," << peso << "," << mejor.confianza << "\n";

        peso_prev=peso; total++;
        if (peso > peso_max) peso_max = peso;
        delta_sum += std::abs(delta);
    }

    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << "Total eventos:    " << total << "\n";
    std::cout << "Peso máximo:      " << peso_max << "t\n";
    std::cout << "Delta promedio:   " << std::fixed << std::setprecision(1)
              << (total > 0 ? delta_sum / total : 0.0) << "t\n";

    csv.close();
    std::cout << "✅ Guardado: cargas_cpp.csv\n";
    return 0;
}
