#pragma once

namespace SymbolicRegression::Utils
{
	template<typename T>
	std::ofstream &operator<<(std::ofstream &ofs, const std::vector<T> &vec)
	{
		ofs << "[";
		for (const auto &element : vec)
		{
			ofs << element << ',';
		}
		ofs << "]";
		return ofs;
	}

	class Log
	{
		std::ofstream os;
		volatile uint64_t id;
	public:
		Log(const char *logFile)
			: os(logFile)
		{
		}

		~Log()
		{
			os.close();
		}

		template<typename...T>
		void print(auto file, auto line, const char *message, T...values)
		{
			os << std::setprecision(20) << "[" << id << " " << file << "(" << line << ")]: " << message << " ";
			((os << values << " "), ...);
			os << std::endl;
			id++;
		}
	};

	inline static Log log("./log.txt");
}
