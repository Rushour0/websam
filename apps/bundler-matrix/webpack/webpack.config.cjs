'use strict';

const path = require('node:path');

/** @type {import('webpack').Configuration} */
module.exports = {
  mode: 'production',
  target: 'web',
  entry: './src/main.ts',
  devtool: false,
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'main.js',
    clean: true,
  },
};
