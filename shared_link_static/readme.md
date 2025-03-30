
you can use following command to show compiler info.

```bash
readelf -p .comment build/main

String dump of section '.comment':
  [     0]  GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
  [    2b]  Ubuntu clang version 18.1.3 (1ubuntu1)
```