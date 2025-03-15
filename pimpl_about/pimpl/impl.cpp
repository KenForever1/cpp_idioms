
#include "impl.h"

Book::Book(): m_p(new BookImpl())
{
}

Book::~Book()
{
  delete m_p;
}

void Book::print()
{
  m_p->print();
}

/* then BookImpl functions */

void Book::BookImpl::print()
{
  std::cout << "print from BookImpl" << std::endl;
  std::cout << "print from BookImpl" << m_Contents << std::endl;
  std::cout << "print from BookImpl lable " << m_label << std::endl;
}