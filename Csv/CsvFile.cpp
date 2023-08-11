#include "CsvFile.h"
#include <iostream>

CsvFile::CsvFile(const char* path)
    : mFileName(path)
{
    std::ifstream is(path);

    if (is)
    {
        std::string line;
        std::getline(is, line);
        std::vector<std::string> buff;
        Row row;
        size_t rowSize = 0;
        if (ParseLine(line, row, buff))
        {
            rowSize = row.size();
            mSamples.push_back(row);
            mTitle.resize(rowSize);
        }
        else
        {
            SplitString(line, mTitle);
            rowSize = mTitle.size();
        }

        while (!is.eof())
        {
            std::getline(is, line);
            if (line.empty())
                break;
            if (!ParseLine(line, row, buff) || row.size() != rowSize)
            {
                std::cerr << "error: read csv error" << std::endl;
                break;
            }
            mSamples.push_back(row);
        }
        is.close();
    }
}

void CsvFile::SplitString(const std::string& line, std::vector<std::string>& result) const
{
    result.clear();
    std::istringstream iss(line);
    for (std::string s; iss >> s;)
        result.push_back(s);
}

bool CsvFile::ParseLine(const std::string& line, Row& row, std::vector<std::string>& buff) const
{
    SplitString(line, buff);
    row.resize(buff.size());
    for (size_t i = 0; i < buff.size(); i++)
    {
        if (!ParseVal(buff[i], row[i]))
            return false;
    }
    return true;
}
