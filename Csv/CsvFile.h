#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <charconv>

class CsvFile
{
    using Row = std::vector<double>;

public:
    explicit CsvFile(const char *path);

    CsvFile() = delete;
    ~CsvFile() = default;

    CsvFile(const CsvFile &) = delete;
    CsvFile(CsvFile &&) noexcept = delete;
    CsvFile &operator=(const CsvFile &) = delete;
    CsvFile &operator=(CsvFile &&) noexcept = delete;

    auto FileName() const noexcept
    {
        return mFileName;
    }

    auto ColumnsCount() const noexcept
    {
        return (uint32_t)mTitle.size();
    }

    auto RowsCount() const noexcept
    {
        return mSamples.size();
    }

    const Row &operator[](size_t idx) const noexcept
    {
        return mSamples[idx];
    }

private:
    inline bool ParseVal(const std::string &strVal, double &val) const noexcept
    {
        return std::errc() == std::from_chars(strVal.data(), strVal.data() + strVal.size(), val).ec;
    }

    bool ParseLine(const std::string &line, std::vector<double> &values, std::vector<std::string> &buff) const;
    void SplitString(const std::string &line, std::vector<std::string> &result) const;

    std::string mFileName;
    std::vector<std::string> mTitle;
    std::vector<Row> mSamples;
};
