/* private.h */
#include "book.h"

#include <iostream>
class Book::BookImpl
{
public:
  void print();
private:
  std::string  m_Contents = "hello world";
  std::string  m_Title;

  std::string m_label = "hhh";
};