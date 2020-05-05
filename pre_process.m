clc
clear all
close all
A = double(imread('x3.jpg'));
A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3); % The number to be given as preprocessing sttement X to the program

%writes to ASCII FILE FOR INPUT TO C/CUDA program 
fid = fopen('input.txt','w');
for i=1:size(X,1)
    fprintf(fid,'%e\t',X(i,:));
    fprintf(fid,'\n');
     if mod(i,100000)==0
         i
    end
    
end
fclose(fid);

%Writes to binary file for input to MPI_Program 
X_s=reshape(X',[],1);
fileID=fopen('input.bin','w');
fwrite(fileID,X_s,'double');
fclose(fileID);