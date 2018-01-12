% close all;clear;clc
addpath('C:\Users\leank\Desktop\TexRelated\matlab2tikz-master\matlab2tikz-master\src')

%    BL = zeros(8,1);
    NBL = zeros(8,1);
 for j = 2 :2 :8
%     if (j == 6)
%      continue;
%     end
%     formatSpec = 'C:/Users/leank/Dropbox/Parallel/Source/Times60K/Blocking/4_node_%d_procs/BL_4_%d_8.txt' ;
       formatSpec = 'C:/Users/leank/Dropbox/Parallel/Source/Times60K/Non_Blocking/4_node_%d_procs/NBL_4_%d_8.txt' ;
   str = sprintf(formatSpec,j,j); 

  C_1= textscan(fopen(str), '%t');
 temp = C_1{1}(2);
%      BL(j) =  str2double(temp);
     NBL(j) =  str2double(temp);

  end
%}

x = [2,4,6,8];
 bl= [BL(2) NBL(2);BL(4) NBL(4);BL(6) NBL(6);BL(8) NBL(8)];
%  nbl= [NBL(2),NBL(4),NBL(8)];

%  hold on

  bar(x,bl);
%  bar(x,nbl);
%  hold off
 xlabel('number of procs')
 ylabel('time')
 title('4 Nodes 60K Blocking Vs Non Blocking 8 threads for 2,4,6,8 procs')
 l = legend('$BL$', '$ NBL $','Location','northeast');
set(l, 'Interpreter', 'Latex', 'FontSize', 11);
set(gca, 'FontSize', 11)
%  matlab2tikz('BL_1.tex');
 
 %}
 
 %{
%  plot(x,sum)
 bar(sum)
 xlabel('number of threads')
 ylabel('time ')
 title('Blocking Performance 1 node 2-8 procs')
 matlab2tikz('Bar_Sum_1.tex');
 
 %---Colors for the graphs---
cc = [
        230, 25, 75;
        60, 180, 75;
        240, 50, 230;
        0, 130, 200;
        245, 130, 48;
        145, 30, 180;
        70, 240, 240;
        210, 245, 60;
        255, 225, 25
    ] / 255;
%---Markers for the graphs---

 
 
 %}
 
 
 
 
 
 
 
 
 
 