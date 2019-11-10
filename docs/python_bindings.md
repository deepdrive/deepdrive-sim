# Building the python bindings 

```
install-local-bindings.sh
```

# Pushing bindings to PyPi

Pushing to the release branch causes CI to publish wheels to the PyPi cheese shop.

```
git push origin master && git push origin master:release
```
