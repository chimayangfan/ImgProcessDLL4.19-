#ifndef __THREAD_POOL_EXECUTOR__
#define __THREAD_POOL_EXECUTOR__

#include "Thread.h"
#include <set>
#include <list>
#include <windows.h>

class CThreadPoolExecutor
{
public:
	CThreadPoolExecutor(void);
	~CThreadPoolExecutor(void);

	/**
	初始化线程池，创建minThreads个线程
	**/
	bool Init(unsigned int minThreads, unsigned int maxThreads, unsigned int maxPendingTaskse);

	/**
	执行任务，若当前任务列表没有满，将此任务插入到任务列表，返回true
	若当前任务列表满了，但当前线程数量小于最大线程数，将创建新线程执行此任务，返回true
	若当前任务列表满了，但当前线程数量等于最大线程数，将丢弃此任务，返回false
	**/
	bool Execute(Runnable * pRunnable, int aa);

	/**
	终止线程池，先制止塞入任务，
	然后等待直到任务列表为空，
	然后设置最小线程数量为0，
	等待直到线程数量为空，
	清空垃圾堆中的任务
	**/
	void Terminate();

	/**
	返回线程池中当前的线程数量
	**/
	unsigned int GetThreadPoolSize();

private:
	/**
	获取任务列表中的任务，若任务列表为空，返回NULL
	**/
	Runnable * GetTask();

	static unsigned int WINAPI StaticThreadFunc(void * arg);

private:
	class CWorker : public CThread
	{
	public:
		CWorker(CThreadPoolExecutor * pThreadPool, Runnable * pFirstTask = NULL);
		~CWorker();
		void Run();
		void Test();

	private:
		CThreadPoolExecutor * m_pThreadPool;
		Runnable * m_pFirstTask;
		volatile bool m_bRun;
	};

	typedef std::set<CWorker *> ThreadPool;
	typedef std::list<Runnable *> Tasks;
	typedef Tasks::iterator TasksItr;
	typedef ThreadPool::iterator ThreadPoolItr;

	ThreadPool m_ThreadPool;
	ThreadPool m_TrashThread;
	Tasks m_Tasks;

	CRITICAL_SECTION m_csTasksLock;
	CRITICAL_SECTION m_csThreadPoolLock;

	volatile bool m_bRun;
	volatile bool m_bEnableInsertTask;
	volatile unsigned int m_minThreads;
	volatile unsigned int m_maxThreads;
	volatile unsigned int m_maxPendingTasks;
};

#endif