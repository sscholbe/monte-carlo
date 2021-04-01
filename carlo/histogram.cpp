#include <vector>
#include <iomanip>
#include <array>
#include <Unknwn.h>
#include <objidl.h>
#include <string>
#include <Windows.h>
#include <GdiPlus.h>
#include <gdipluspixelformats.h>
#include <sstream>
#include <numeric>
#undef min
#undef max

#pragma comment(lib, "gdiplus.lib")

inline void measure_string(Gdiplus::Graphics *g, const std::wstring &text, SIZE *size) {
    HDC hdc = g->GetHDC();
    GetTextExtentPoint32W(hdc, text.c_str(), text.length(), size);
    g->ReleaseHDC(hdc);
}

unsigned find_suitable_max_bound(unsigned n) {
    std::array<double, 10> scale = {1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 6, 8};
    for (unsigned i = 1; ; i *= 10) {
        for (double s : scale) {
            unsigned m = (int) (s * i * 10);
            if (m >= n) {
                return m;
            }
        }
    }
}

template<class _First, class _Last>
double stable_mean(_First first, _Last last) {
    double mu = 0;
    size_t i = 0;
    for (_First cur = first; cur != last; cur++, i++) {
        mu += (*cur - mu) / (i + 1);
    }
    return mu;
}

void create_histogram(std::vector<double> &losses, const std::wstring &sub) {
    std::sort(losses.begin(), losses.end());

    double expected_loss = stable_mean(losses.begin(), losses.end());
    double var95 = losses[(int) (losses.size() * 0.95)];
    double var99 = losses[(int) (losses.size() * 0.99)];
    double es95 = stable_mean(losses.begin() += (int) (losses.size() * 0.95), losses.end());
    double es99 = stable_mean(losses.begin() += (int) (losses.size() * 0.99), losses.end());

    const int bin_size = 10'000'000, number_of_bins = 30;

    std::array<unsigned, number_of_bins> bins = {0};
    for (double loss : losses) {
        if (loss < bin_size * number_of_bins) {
            bins[(unsigned) (loss / bin_size)]++;
        }
    }

    unsigned width = 16 * number_of_bins + 60 * 2, height = 500;
    std::vector<uint8_t> imageData(width * height * 4);

    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;

    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    Gdiplus::Font myFont(L"MS Sans Serif", 8);
    Gdiplus::Font bold(L"MS Sans Serif", 8, Gdiplus::FontStyleBold);

    Gdiplus::SolidBrush black(Gdiplus::Color::Black), gray(Gdiplus::Color::Gray);

    Gdiplus::Bitmap bmp(width, height, 4 * width, PixelFormat32bppARGB, imageData.data());
    Gdiplus::Graphics *g = Gdiplus::Graphics::FromImage(&bmp);

    g->SetTextRenderingHint(Gdiplus::TextRenderingHintSingleBitPerPixelGridFit);
    g->Clear(Gdiplus::Color::White);

    SIZE size;

    std::wostringstream title;
    title << L"Loss distribution (" << std::fixed;
    if (losses.size() < 10'000) {
        title << losses.size();
    } else if (losses.size() < 1'000'000) {
        title << (int) (losses.size() / 1'000) << "k";
    } else {
        title << std::setprecision(2) << (losses.size() / 1'000'000) << "M";
    }
    title << " iterations)";

    measure_string(g, title.str(), &size);
    g->DrawString(title.str().c_str(), -1, &bold,
        Gdiplus::PointF(width / 2 - (float) size.cx / 2, 20), &black);

    g->TranslateTransform(0, 70);

    unsigned max_bin = find_suitable_max_bound(*std::max_element(bins.begin(), bins.end()));

    g->TranslateTransform(60, 0);

    std::wostringstream desc;
    desc << "Expected loss" << std::endl;
    desc << "VaR 95%" << std::endl;
    desc << "VaR 99%" << std::endl;
    desc << "ES 95%" << std::endl;
    desc << "ES 99%" << std::endl;

    std::wostringstream vals;
    vals << std::fixed << std::setprecision(2);
    vals << expected_loss / 1E6 << std::endl;
    vals << var95 / 1E6 << std::endl;
    vals << var99 / 1E6 << std::endl;
    vals << es95 / 1E6 << std::endl;
    vals << es99 / 1E6 << std::endl;

    Gdiplus::StringFormat rtl(Gdiplus::StringFormatFlagsDirectionRightToLeft);

    g->DrawString(desc.str().c_str(), -1, &myFont,
        Gdiplus::PointF(number_of_bins * 16 - 160, 10), &black);
    g->DrawString(vals.str().c_str(), -1, &myFont,
        Gdiplus::PointF(number_of_bins * 16 - 10, 10), &rtl, &black);

    Gdiplus::Pen pen(&gray, 1);

    for (unsigned i = 0; i <= 10; i++) {
        std::wstring ws = std::to_wstring(max_bin / 10 * (10 - i));
        measure_string(g, ws, &size);
        g->DrawString(ws.c_str(), -1, &myFont,
            Gdiplus::PointF((float) -size.cx - 10, i * 30 - (float) size.cy / 2), &black);
        g->DrawLine(&pen, Gdiplus::Point(-6, i * 30), Gdiplus::Point(0, i * 30));
    }
    g->DrawRectangle(&pen, Gdiplus::Rect(0, 0, number_of_bins * 16, 300));

    g->DrawString(L"Frequency", -1, &myFont, Gdiplus::PointF(-50, -30), &black);

    Gdiplus::SolidBrush blue(Gdiplus::Color::Blue);
    for (size_t i = 0; i < bins.size(); i++) {
        int height = (int) ((bins[i] / (float) max_bin) * 300);
        g->FillRectangle(&blue, i * 16, 300 - height, 16, height);
    }

    g->TranslateTransform(0, 308);

    for (size_t i = 0; i <= bins.size(); i += 2) {
        std::wstring ws = std::to_wstring(i * 10);
        measure_string(g, ws, &size);
        g->DrawString(ws.c_str(), -1, &myFont,
            Gdiplus::PointF(i * 16 - (float) size.cx / 2, 0), &black);

        g->DrawLine(&pen, Gdiplus::Point(i * 16, -8), Gdiplus::Point(i * 16, -2));
    }

    g->TranslateTransform(0, 16);

    measure_string(g, L"Loss in CHFm", &size);
    g->DrawString(L"Loss in CHFm", -1, &myFont,
        Gdiplus::PointF(number_of_bins * 16 / 2 - (float) size.cx / 2, 0), &black);

    g->TranslateTransform(0, 24);
    g->DrawString(sub.c_str(), -1, &myFont, Gdiplus::PointF(), &black);

    CLSID pngClsid;
    CLSIDFromString(L"{557CF406-1A04-11D3-9A73-0000F81EF32E}", &pngClsid);
    bmp.Save(L"out/histogram.png", &pngClsid, NULL);
}
