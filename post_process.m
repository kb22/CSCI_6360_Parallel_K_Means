clc
clear all
close all

%Original image
A = double(imread('x3.jpg'));
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1
img_size = size(A);

%Image from C program
X= importdata('output.txt');
A2 = reshape(X, img_size(1), img_size(2), 3);


%Display original image
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image
subplot(1, 2, 2);
imagesc(A2)
title(sprintf('Compressed image'));
